import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os,logging
from .evaluator import evaluate
from utils.general_utils import save_checkpoint


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]", leave=False)
    #_has_logging.infoed_grad_debug = False

    for batch in pbar:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        rgb = batch['rgb']

        optimizer.zero_grad()
        outputs = model(rgb)

        # Support criterion that returns either dict or (total_loss, dict)
        crit_out = criterion(outputs, batch)
        if isinstance(crit_out, tuple) or isinstance(crit_out, list):
            total_loss, loss_dict = crit_out[0], crit_out[1]
        elif isinstance(crit_out, dict):
            loss_dict = crit_out
            total_loss = loss_dict.get('total_loss')
            if total_loss is None:
                raise ValueError("criterion returned dict but no 'total_loss' key found.")
        else:
            raise ValueError("criterion must return dict or (total_loss, dict).")

        total_loss.backward()

        # if not _has_logging.infoed_grad_debug and epoch == 0:
        #     logging.info("\n--- [ONE-TIME DEBUG] Gradient Check ---")
        #     for name, p in model.named_parameters():
        #         # 我们只关心和分割任务直接相关的部分
        #         if 'predictor_seg' in name or 'projector_s' in name or 'projector_p_seg' in name:
        #             is_grad_none = p.grad is None
        #             grad_norm = "N/A" if is_grad_none else p.grad.norm().item()
        #             logging.info(
        #                 f"{name:<50} requires_grad={p.requires_grad}, grad_is_None={is_grad_none}, grad_norm={grad_norm}")
        #     logging.info("--- End of Gradient Check ---\n")
        #     _has_logging.infoed_grad_debug = True

        optimizer.step()

        total_train_loss += float(total_loss.item())
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    logging.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, config, device, checkpoint_dir='checkpoints'):
    best_val_metric = float('inf')

    for epoch in range(config['training']['epochs']):
        logging.info(f"\n----- Starting Epoch {epoch + 1}/{config['training']['epochs']} -----")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # --- Scheduler Advice ---
        # If using StepLR (current):
        scheduler.step()
        # If you switch to ReduceLROnPlateau in the config, change the call to:
        # scheduler.step(val_metrics['val_loss'])

        is_best = val_metrics['depth_rmse'] < best_val_metric
        if is_best:
            best_val_metric = val_metrics['depth_rmse']
            logging.info(f"  -> New best model found with Depth RMSE: {best_val_metric:.4f}")

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_metric': best_val_metric,
        }, is_best, checkpoint_dir=checkpoint_dir)

    logging.info("\n----- Training Finished -----")
    logging.info(f"Best model saved with Depth RMSE: {best_val_metric:.4f}")