import torch
import torch.nn as nn
#from .hsic import HSIC
from .linear_cka import LinearCKA
import torch.nn.functional as F
from metrics.lpips import LPIPSMetric

def _sobel(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = kx.transpose(2,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy

def seg_edge_consistency_loss(seg_logits, geom_from_zs, weight=0.1, tau=0.1):
    """
    让分割的边缘（p_max 的梯度）和几何的边缘对齐。
    seg_logits: [B,C,H,W]
    geom_from_zs: [B,1,H,W]
    """
    # 对齐分辨率
    if seg_logits.shape[-2:] != geom_from_zs.shape[-2:]:
        geom_from_zs = F.interpolate(geom_from_zs, size=seg_logits.shape[-2:], mode='bilinear', align_corners=False)

    # 分割边缘：对 softmax 后的最大类概率求梯度
    p = torch.softmax(seg_logits, dim=1)
    p_max = p.max(dim=1, keepdim=True).values
    gx_p, gy_p = _sobel(p_max)

    # 几何边缘（不反传给几何，避免干扰 z_s 重构）
    g = (geom_from_zs - geom_from_zs.mean()) / (geom_from_zs.std() + 1e-6)
    gx_g, gy_g = _sobel(g.detach())

    # 归一化 & 只在强几何边缘处计算（掩码）
    def _norm(a):
        return a / (a.abs().mean() + 1e-6)
    gx_p, gy_p = _norm(gx_p), _norm(gy_p)
    gx_g, gy_g = _norm(gx_g), _norm(gy_g)

    edge_mag = (gx_g.abs() + gy_g.abs())
    mask = (edge_mag > tau).float()  # 只在强边缘处约束

    l = ( (gx_p - gx_g).abs() + (gy_p - gy_g).abs() ) * mask
    return weight * l.mean()

def edge_consistency_loss(geom_from_zs, depth_gt, weight=0.1):
    """
    让 z_s 的几何重构在边缘上和 GT 深度一致。输入都是 [B,1,H,W]。
    """
    # 尺寸对齐
    if geom_from_zs.shape[-2:] != depth_gt.shape[-2:]:
        depth_gt = F.interpolate(depth_gt, size=geom_from_zs.shape[-2:], mode="bilinear", align_corners=False)

    # 归一化，避免尺度影响梯度
    g = (geom_from_zs - geom_from_zs.mean()) / (geom_from_zs.std() + 1e-6)
    d = (depth_gt     - depth_gt.mean())     / (depth_gt.std()     + 1e-6)

    gx_g, gy_g = _sobel(g)
    gx_d, gy_d = _sobel(d)
    l = (gx_g - gx_d).abs().mean() + (gy_g - gy_d).abs().mean()
    return weight * l


class EdgeConsistencyLoss(nn.Module):
    """
    让 Geometry-from-zs 的边缘/梯度和 GT 深度的边缘一致。
    返回未加权的标量；权重在 CompositeLoss.total_loss 中统一乘。
    """
    def __init__(self):
        super().__init__()
        # 注册 Sobel 核为 buffer，自动随模型搬到对应设备/精度
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(2,3).contiguous()
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def _normalize(self, x: torch.Tensor, eps: float = 1e-6):
        # 每张图做简单 z-score，避免尺度影响（允许对 geom_pred 反传梯度）
        m = x.mean(dim=(2,3), keepdim=True)
        s = x.std (dim=(2,3), keepdim=True)
        return (x - m) / (s + eps)

    def _grads(self, x: torch.Tensor):
        gx = F.conv2d(x, self.sobel_x.to(x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.dtype), padding=1)
        return gx, gy

    def forward(self, geom_pred: torch.Tensor, depth_gt: torch.Tensor) -> torch.Tensor:
        """
        geom_pred: [B,1,H,W] （你的 Geometry-from-zs probe：outputs['recon_geom']）
        depth_gt : [B,1,H_gt,W_gt] 或 [B,H_gt,W_gt]
        """
        # 对齐分辨率到 geom_pred
        if depth_gt.dim() == 3:
            depth_gt = depth_gt.unsqueeze(1)
        if geom_pred.dim() == 3:
            geom_pred = geom_pred.unsqueeze(1)
        if depth_gt.shape[-2:] != geom_pred.shape[-2:]:
            depth_gt = F.interpolate(depth_gt, size=geom_pred.shape[-2:], mode="bilinear", align_corners=False)

        # 归一化到相近尺度
        g = self._normalize(geom_pred)
        d = self._normalize(depth_gt)

        # 取 Sobel 梯度并做 L1 差
        gx_g, gy_g = self._grads(g)
        gx_d, gy_d = self._grads(d)
        loss = (gx_g - gx_d).abs().mean() + (gy_g - gy_d).abs().mean()
        return loss

class CompositeLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.weights = loss_weights

        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.depth_loss = nn.L1Loss()
        self.scene_loss = nn.CrossEntropyLoss()

        #self.independence_loss = HSIC(normalize=True)
        self.independence_loss = LinearCKA(eps=1e-6)
        self.recon_geom_loss = nn.L1Loss()
        #self.recon_app_loss = nn.MSELoss()
        self.recon_app_loss_lpips = LPIPSMetric(net='vgg')  # LPIPS部分
        self.recon_app_loss_l1 = nn.L1Loss()  # 新增L1部分
        self.edge_consistency_loss = EdgeConsistencyLoss()
        self.current_epoch = 0
        self.ind_warmup_epochs = int(loss_weights.get('ind_warmup_epochs', 10))
        self.lambda_ind_base = float(loss_weights.get('lambda_independence', 0.1))

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def forward(self, outputs, targets):
        loss_dict = {}

        # --- 1. Task Losses ---
        l_seg = self.seg_loss(outputs['pred_seg'], targets['segmentation'])
        l_depth = self.depth_loss(outputs['pred_depth'], targets['depth'])
        l_scene = self.scene_loss(outputs['pred_scene'], targets['scene_type'])

        l_task = (self.weights.get('lambda_seg', 1.0) * l_seg +
                  self.weights.get('lambda_depth', 1.0) * l_depth +
                  self.weights.get('lambda_scene', 1.0) * l_scene)

        loss_dict.update({'task_loss': l_task, 'seg_loss': l_seg, 'depth_loss': l_depth, 'scene_loss': l_scene})

        # --- 2. Causal Independence Loss ---
        z_s = outputs['z_s']
        z_p_seg = outputs['z_p_seg']
        z_p_depth = outputs['z_p_depth']
        z_p_scene = outputs['z_p_scene']

        z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)

        loss_dict['cka_seg'] = self.independence_loss(z_s_centered, z_p_seg - z_p_seg.mean(0, keepdim=True))
        loss_dict['cka_depth'] = self.independence_loss(z_s_centered, z_p_depth - z_p_depth.mean(0, keepdim=True))
        loss_dict['cka_scene'] = self.independence_loss(z_s_centered, z_p_scene - z_p_scene.mean(0, keepdim=True))

        l_ind = (self.independence_loss(z_s_centered, z_p_seg - z_p_seg.mean(0, keepdim=True)) +
                 self.independence_loss(z_s_centered, z_p_depth - z_p_depth.mean(0, keepdim=True)) +
                 self.independence_loss(z_s_centered, z_p_scene - z_p_scene.mean(0, keepdim=True)))
        loss_dict['independence_loss'] = l_ind

        # --- 3. Reconstruction Loss (Main and Auxiliary) ---
        # Main reconstruction loss
        l_recon_g = self.recon_geom_loss(outputs['recon_geom'], targets['depth'])
        #l_recon_a = self.recon_app_loss(outputs['recon_app'], targets['appearance_target'])
        l_recon_a_lpips = self.recon_app_loss_lpips(outputs['recon_app'], targets['appearance_target'])
        l_recon_a_l1 = self.recon_app_loss_l1(outputs['recon_app'], targets['appearance_target'])

        # 混合损失
        l_recon_a = l_recon_a_lpips + self.weights.get('lambda_l1_recon', 1.0) * l_recon_a_l1
        loss_dict.update({'recon_geom_loss': l_recon_g, 'recon_app_loss': l_recon_a})

        # --- START: CORRECTED AUXILIARY LOSS CALCULATION ---
        recon_geom_aux = outputs['recon_geom_aux']
        recon_app_aux = outputs['recon_app_aux']

        aux_size_g = recon_geom_aux.shape[2:]
        aux_size_a = recon_app_aux.shape[2:]

        target_depth_aux = F.interpolate(targets['depth'], size=aux_size_g, mode='bilinear', align_corners=False)
        target_app_aux = F.interpolate(targets['appearance_target'], size=aux_size_a, mode='bilinear',
                                       align_corners=False)

        l_recon_g_aux = self.recon_geom_loss(recon_geom_aux, target_depth_aux)

        # 让辅助外观损失也使用混合模式
        l_recon_a_lpips_aux = self.recon_app_loss_lpips(recon_app_aux, target_app_aux)
        l_recon_a_l1_aux = self.recon_app_loss_l1(recon_app_aux, target_app_aux)
        l_recon_a_aux = l_recon_a_lpips_aux + self.weights.get('lambda_l1_recon', 1.0) * l_recon_a_l1_aux

        loss_dict.update({'recon_geom_aux_loss': l_recon_g_aux, 'recon_app_aux_loss': l_recon_a_aux})

        l_depth_from_zp = self.depth_loss(outputs['pred_depth_from_zp'], targets['depth'])
        loss_dict['depth_from_zp_loss'] = l_depth_from_zp
        if 'recon_geom' in outputs and 'depth' in targets:
            l_edge = self.edge_consistency_loss(outputs['recon_geom'], targets['depth'])
        else:
            l_edge = torch.tensor(0.0, device=outputs['pred_depth'].device)
        loss_dict['edge_consistency_loss'] = l_edge
        edge_w = self.weights.get('alpha_recon_geom_edges', 0.1)
        seg_edge_w = self.weights.get('beta_seg_edge_from_geom', 0.1)
        # --- 4. Total Loss ---
        total_loss = ((l_task +
                      #self.weights.get('lambda_independence', 0) * l_ind +
                      self.weights.get('alpha_recon_geom', 0) * l_recon_g +
                      self.weights.get('beta_recon_app', 0) * l_recon_a +
                      self.weights.get('alpha_recon_geom_aux', 0) * l_recon_g_aux +
                      self.weights.get('beta_recon_app_aux', 0) * l_recon_a_aux+
                      self.weights.get('lambda_depth_zp', 0) * l_depth_from_zp)+
                      self.weights.get('lambda_edge_consistency', 0.0) * l_edge+
                      edge_consistency_loss(outputs['recon_geom'], targets['depth'], weight=edge_w)+
                      seg_edge_consistency_loss(outputs['pred_seg'],outputs['recon_geom'],weight=seg_edge_w))

        warmup_ratio = min(1.0, self.current_epoch / max(1, self.ind_warmup_epochs))
        lambda_ind = self.lambda_ind_base * warmup_ratio
        total_loss = total_loss + lambda_ind * l_ind
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

class AdaptiveCompositeLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.weights = loss_weights

        # ==== 基础项 ====
        self.seg_loss   = nn.CrossEntropyLoss(ignore_index=255)
        self.depth_loss = nn.L1Loss()
        self.scene_loss = nn.CrossEntropyLoss()

        self.independence_loss = LinearCKA(eps=1e-6)

        self.recon_geom_loss     = nn.L1Loss()
        self.recon_app_loss_lpips= LPIPSMetric(net='vgg')
        self.recon_app_loss_l1   = nn.L1Loss()

        self.edge_consistency_loss = EdgeConsistencyLoss()

        # ==== 用 YAML 权重做 log_var 的先验初始化（log_var=log(1/λ)） ====
        def _to_logvar(inv_weight_default_one):
            # 期望初始权重 ~ λ => precision ≈ λ => log_var ≈ log(1/λ)
            lam = float(inv_weight_default_one)
            lam = max(lam, 1e-6)
            return math.log(1.0 / lam)

        import math
        init = {
            'seg'        : _to_logvar(self.weights.get('lambda_seg',       1.0)),
            'depth'      : _to_logvar(self.weights.get('lambda_depth',     1.0)),
            'scene'      : _to_logvar(self.weights.get('lambda_scene',     1.0)),
            'ind'        : _to_logvar(self.weights.get('lambda_independence', 1.0)),
            'recon_geom' : _to_logvar(self.weights.get('alpha_recon_geom', 1.0)),
            'recon_app'  : _to_logvar(self.weights.get('beta_recon_app',   1.0)),
        }
        self.log_vars = nn.ParameterDict({
            k: nn.Parameter(torch.tensor([v], dtype=torch.float32))
            for k, v in init.items()
        })

        # 轻度范围，避免 early collapse / explosion
        self._logvar_min = -4.0
        self._logvar_max =  4.0

    def _uw(self, name: str, loss_scalar: torch.Tensor) -> torch.Tensor:
        """
        Kendall 不确定性加权：
        0.5 * exp(-log_var) * L + 0.5 * log_var
        同时对 log_var 做轻度 clamp 保稳。
        """
        lv = self.log_vars[name]
        lv_clamped = torch.clamp(lv, self._logvar_min, self._logvar_max)
        return 0.5 * torch.exp(-lv_clamped) * loss_scalar + 0.5 * lv_clamped

    def forward(self, outputs, targets):
        loss_dict = {}

        # ===== 1) 主任务 =====
        l_seg   = self.seg_loss(outputs['pred_seg'],   targets['segmentation'])
        l_depth = self.depth_loss(outputs['pred_depth'], targets['depth'])
        l_scene = self.scene_loss(outputs['pred_scene'], targets['scene_type'])

        loss_seg   = self._uw('seg',   l_seg)
        loss_depth = self._uw('depth', l_depth)
        loss_scene = self._uw('scene', l_scene)

        l_task = loss_seg + loss_depth + loss_scene
        loss_dict.update({
            'seg_loss':   l_seg,    # Tensor
            'depth_loss': l_depth,  # Tensor
            'scene_loss': l_scene,  # Tensor
            'task_loss':  l_task,   # Tensor
        })

        # ===== 2) 解耦独立性 =====
        z_s       = outputs['z_s']
        z_p_seg   = outputs['z_p_seg']
        z_p_depth = outputs['z_p_depth']
        z_p_scene = outputs['z_p_scene']

        z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)

        cka_seg   = self.independence_loss(z_s_centered, z_p_seg   - z_p_seg.mean(0,   keepdim=True))
        cka_depth = self.independence_loss(z_s_centered, z_p_depth - z_p_depth.mean(0, keepdim=True))
        cka_scene = self.independence_loss(z_s_centered, z_p_scene - z_p_scene.mean(0, keepdim=True))

        l_ind = cka_seg + cka_depth + cka_scene
        loss_ind = self._uw('ind', l_ind)

        # ✅ 恢复三个 CKA 明细，避免 evaluator 回退 0.0
        loss_dict['cka_seg']   = cka_seg
        loss_dict['cka_depth'] = cka_depth
        loss_dict['cka_scene'] = cka_scene
        loss_dict['independence_loss'] = l_ind

        # ===== 3) 重建（主） =====
        l_recon_g        = self.recon_geom_loss(outputs['recon_geom'], targets['depth'])
        l_recon_a_lpips  = self.recon_app_loss_lpips(outputs['recon_app'], targets['appearance_target'])
        l_recon_a_l1     = self.recon_app_loss_l1(outputs['recon_app'], targets['appearance_target'])
        l_recon_a        = l_recon_a_lpips + self.weights.get('lambda_l1_recon', 1.0) * l_recon_a_l1

        loss_recon_g = self._uw('recon_geom', l_recon_g)
        loss_recon_a = self._uw('recon_app',  l_recon_a)

        loss_dict.update({
            'recon_geom_loss': l_recon_g,
            'recon_app_loss':  l_recon_a,
        })

        # ===== 4) 重建（辅助，静态） =====
        recon_geom_aux = outputs['recon_geom_aux']
        recon_app_aux  = outputs['recon_app_aux']
        aux_size_g = recon_geom_aux.shape[2:]
        aux_size_a = recon_app_aux.shape[2:]

        target_depth_aux = F.interpolate(targets['depth'], size=aux_size_g, mode='bilinear', align_corners=False)
        target_app_aux   = F.interpolate(targets['appearance_target'], size=aux_size_a, mode='bilinear', align_corners=False)

        l_recon_g_aux       = self.recon_geom_loss(recon_geom_aux, target_depth_aux)
        l_recon_a_lpips_aux = self.recon_app_loss_lpips(recon_app_aux, target_app_aux)
        l_recon_a_l1_aux    = self.recon_app_loss_l1(recon_app_aux, target_app_aux)
        l_recon_a_aux       = l_recon_a_lpips_aux + self.weights.get('lambda_l1_recon', 1.0) * l_recon_a_l1_aux

        loss_dict.update({
            'recon_geom_aux_loss': l_recon_g_aux,
            'recon_app_aux_loss':  l_recon_a_aux,
        })

        # ===== 5) 其他一致性项（静态） =====
        l_depth_from_zp = self.depth_loss(outputs['pred_depth_from_zp'], targets['depth'])
        l_edge          = self.edge_consistency_loss(outputs['recon_geom'], targets['depth'])
        edge_w     = self.weights.get('alpha_recon_geom_edges', 0.1)
        seg_edge_w = self.weights.get('beta_seg_edge_from_geom', 0.1)

        loss_dict.update({
            'depth_from_zp_loss':     l_depth_from_zp,
            'edge_consistency_loss':  l_edge,
        })

        # ===== 6) 汇总（删除“重复的 edge_consistency 直加项”） =====
        total_loss = (
            l_task + loss_ind + loss_recon_g + loss_recon_a +
            self.weights.get('alpha_recon_geom_aux', 0.0) * l_recon_g_aux +
            self.weights.get('beta_recon_app_aux',  0.0) * l_recon_a_aux +
            self.weights.get('lambda_depth_zp',     0.0) * l_depth_from_zp +
            self.weights.get('lambda_edge_consistency', 0.0) * l_edge +
            # 只保留这条：seg 边缘一致性（功能性项）
            seg_edge_consistency_loss(outputs['pred_seg'], outputs['recon_geom'], weight=seg_edge_w)
        )

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
