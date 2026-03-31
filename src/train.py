import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils

from src.models import DualEncoderUNet, RGBBaselineUNet, EventWarpNet, EventWarpNetV2, SynthesisUNet
from src.losses import CombinedLoss, CharbonnierLoss
from src.metrics import calculate_psnr, calculate_ssim


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Construct model from config dict and move it to device."""
    model_type = cfg.get('model_type', 'dual')
    if model_type == 'baseline':
        model = RGBBaselineUNet(n_channels=6, n_classes=3)
    elif model_type == 'warp':
        model = EventWarpNet()
    elif model_type == 'warp_v2':
        model = EventWarpNetV2()
    elif model_type == 'synthesis':
        model = SynthesisUNet(in_channels=cfg.get('in_channels', 11))
    else:
        model = DualEncoderUNet(n_channels_rgb=6, n_channels_evt=5, n_classes=3)
    return model.to(device)


def build_criterion(cfg: dict, device: torch.device) -> nn.Module:
    """Construct loss from config dict."""
    loss_type = cfg.get('loss_type', 'combined')
    if loss_type == 'timelens':
        from src.losses import TimeLensLoss
        return TimeLensLoss(
            device=device,
            lambda_perceptual=cfg.get('lambda_perceptual', 0.1),
        ).to(device)
    return CombinedLoss(
        device=device,
        lambda_char=cfg.get('lambda_char', 1.0),
        lambda_perceptual=cfg.get('lambda_perceptual', 0.05),
        lambda_ssim=cfg.get('lambda_ssim', 0.2),
        lambda_smooth=cfg.get('lambda_smooth', 0.01),
        lambda_event_weighted=cfg.get('lambda_event_weighted', 0.0),
    ).to(device)


def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    """Construct optimizer from config dict."""
    opt_type = cfg.get('optimizer_type', 'adamw')
    if opt_type == 'adam':
        return optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-4))
    return optim.AdamW(
        model.parameters(),
        lr=cfg.get('lr', 1e-4),
        weight_decay=cfg.get('weight_decay', 1e-4),
    )


def build_scheduler(optimizer: optim.Optimizer, cfg: dict) -> optim.lr_scheduler.LRScheduler:
    """LinearLR warmup → CosineAnnealingLR via SequentialLR."""
    warmup_epochs = cfg.get('warmup_epochs', 5)
    num_epochs = cfg.get('num_epochs', 80)
    if warmup_epochs == 0:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=cfg.get('lr_min', 1e-6),
        )
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=cfg.get('lr_min', 1e-6),
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    """
    Load the latest checkpoint from a directory (or a specific file).
    Returns (start_epoch, best_psnr).
    """
    if os.path.isdir(path):
        ckpts = sorted(glob.glob(os.path.join(path, 'checkpoint_epoch_*.pth')))
        if not ckpts:
            return 1, 0.0
        path = ckpts[-1]

    print(f'Resuming from {path}')
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_psnr   = ckpt.get('best_psnr', 0.0)
    print(f'  -> Epoch {start_epoch}, best PSNR so far: {best_psnr:.2f} dB')
    return start_epoch, best_psnr


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    best_psnr: float,
    val_psnr: float,
) -> float:
    """Save latest checkpoint, prune old ones, and update best model if improved."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_psnr':            best_psnr,
    }, ckpt_path)

    # Keep only the most recent checkpoint
    for old in sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')))[:-1]:
        os.remove(old)

    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f'  -> New best PSNR: {best_psnr:.2f} dB')

    return best_psnr


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def _forward(model: nn.Module, f0, f1, voxel):
    """
    Unified forward pass.
    - RGBBaselineUNet:  returns (pred,)
    - EventWarpNet:     returns (pred, flow_t0, flow_t1)
    - EventWarpNetV2:   returns (pred, flow_t0, flow_t1, flow_t0_c, flow_t1_c)
    - DualEncoderUNet:  returns (pred,)
    Always returns a tuple so callers unpack consistently.
    """
    if isinstance(model, RGBBaselineUNet):
        return (model(torch.cat([f0, f1], dim=1)),)
    if isinstance(model, SynthesisUNet):
        return (model(torch.cat([f0, voxel, f1], dim=1)),)
    if isinstance(model, EventWarpNetV2):
        return model(torch.cat([f0, f1], dim=1), voxel)  # 5-tuple
    if isinstance(model, EventWarpNet):
        return model(torch.cat([f0, f1], dim=1), voxel)   # 3-tuple
    return (model(torch.cat([f0, f1], dim=1), voxel),)


def _compute_loss(criterion, outputs, gt, f0, voxel=None, f1=None):
    """
    Dispatch loss computation based on what _forward returned.
    Passes flows to CombinedLoss when available.
    For EventWarpNetV2 (5-tuple), adds coarse flow supervision.
    """
    from src.losses import CombinedLoss
    from src.models import backwarp
    pred = outputs[0]

    # EventWarpNetV2: 5-tuple (pred, flow_t0, flow_t1, flow_t0_c, flow_t1_c)
    if len(outputs) == 5 and isinstance(criterion, CombinedLoss):
        _, flow_t0, flow_t1, flow_t0_c, flow_t1_c = outputs
        loss = criterion(pred, gt, flow_t0=flow_t0, flow_t1=flow_t1, f0=f0, voxel=voxel)
        # Coarse flow supervision: photometric loss on coarse warp
        if f1 is not None:
            coarse_warp = 0.5 * backwarp(f0, flow_t0_c) + 0.5 * backwarp(f1, flow_t1_c)
            loss = loss + 0.25 * criterion.char(coarse_warp, gt)
        return loss

    # EventWarpNet: 3-tuple (pred, flow_t0, flow_t1)
    if len(outputs) == 3 and isinstance(criterion, CombinedLoss):
        _, flow_t0, flow_t1 = outputs
        return criterion(pred, gt, flow_t0=flow_t0, flow_t1=flow_t1, f0=f0, voxel=voxel)

    if isinstance(criterion, CombinedLoss):
        return criterion(pred, gt, voxel=voxel)
    return criterion(pred, gt)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def overfit_one_batch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    iters: int = 1000,
) -> float:
    """
    Sanity check: verify the model can memorize a single batch.
    Loss should reach < 0.01 to confirm the architecture is learning.
    """
    model.train()
    f0, f1, voxel, gt = [x.to(device) for x in next(iter(loader))]

    pbar = tqdm(range(iters), desc='Overfitting batch')
    loss_val = 0.0
    for _ in pbar:
        optimizer.zero_grad()
        outputs = _forward(model, f0, f1, voxel)
        loss = _compute_loss(criterion, outputs, gt, f0, voxel=voxel, f1=f1)
        loss.backward()
        # No gradient clipping for overfit sanity check — clipping at 1.0 throttles
        # learning too aggressively on a fresh large network and prevents convergence.
        optimizer.step()
        loss_val = loss.item()
        pbar.set_postfix({'loss': f'{loss_val:.6f}'})

    return loss_val


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer=None,
    max_norm: float = 5.0,
) -> float:
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f'Epoch {epoch}')

    for i, (f0, f1, voxel, gt) in enumerate(pbar):
        f0, f1, voxel, gt = f0.to(device), f1.to(device), voxel.to(device), gt.to(device)

        optimizer.zero_grad()
        amp_enabled = scaler.is_enabled()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = _forward(model, f0, f1, voxel)
            loss = _compute_loss(criterion, outputs, gt, f0, voxel=voxel, f1=f1)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f'  [WARN] NaN/Inf loss at batch {i}, skipping')
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if writer is not None and i % 10 == 0:
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(loader) + i)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'gnorm': f'{grad_norm:.1f}'})

    avg_loss = running_loss / len(loader)
    if writer is not None:
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer=None,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = total_psnr = total_ssim = 0.0

    for i, (f0, f1, voxel, gt) in enumerate(loader):
        f0, f1, voxel, gt = f0.to(device), f1.to(device), voxel.to(device), gt.to(device)
        outputs = _forward(model, f0, f1, voxel)
        pred = outputs[0].clamp(0, 1)

        total_loss += _compute_loss(criterion, outputs, gt, f0, voxel=voxel, f1=f1).item()
        total_psnr += calculate_psnr(pred, gt).item()
        total_ssim += calculate_ssim(pred, gt).item()

        if i == 0 and writer is not None:
            writer.add_image('Val/Prediction',  vutils.make_grid(pred[:4].clamp(0, 1)), epoch)
            writer.add_image('Val/GroundTruth', vutils.make_grid(gt[:4]),               epoch)
            error = (pred - gt).abs() * 5.0
            writer.add_image('Val/ErrorMap_5x', vutils.make_grid(error[:4].clamp(0, 1)), epoch)

    n = len(loader)
    avg_loss = total_loss / n
    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n

    if writer is not None:
        writer.add_scalar('Loss/val',      avg_loss, epoch)
        writer.add_scalar('Metrics/PSNR',  avg_psnr, epoch)
        writer.add_scalar('Metrics/SSIM',  avg_ssim, epoch)

    return avg_loss, avg_psnr, avg_ssim


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict,
    start_epoch: int = 1,
    best_psnr: float = 0.0,
    writer=None,
) -> list[dict]:
    """
    Full training loop. Returns the training log (list of per-epoch dicts).
    Checkpoints and log are saved to cfg['checkpoint_dir'].
    """
    checkpoint_dir = cfg['checkpoint_dir']
    num_epochs     = cfg.get('num_epochs', 40)
    log_path       = os.path.join(checkpoint_dir, 'training_log.json')

    training_log: list[dict] = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            training_log = json.load(f)

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer,
            max_norm=cfg.get('max_norm', 5.0),
        )
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        scheduler.step()

        print(
            f'Epoch {epoch:03d}/{num_epochs} | '
            f'Train {train_loss:.4f} | Val {val_loss:.4f} | '
            f'PSNR {val_psnr:.2f} dB | SSIM {val_ssim:.4f}'
        )

        training_log.append({
            'epoch':      epoch,
            'train_loss': round(train_loss, 6),
            'val_loss':   round(val_loss,   6),
            'val_psnr':   round(val_psnr,   4),
            'val_ssim':   round(val_ssim,   4),
        })
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        best_psnr = save_checkpoint(
            checkpoint_dir, epoch, model, optimizer, scheduler, best_psnr, val_psnr
        )

    return training_log
