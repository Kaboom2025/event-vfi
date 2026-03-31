import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CharbonnierLoss(nn.Module):
    """Charbonnier (smooth L1) loss -- more robust to outliers than MSE."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class PerceptualLoss(nn.Module):
    """
    VGG19 perceptual loss using relu3_3 features (index 18).
    Inputs are expected in [0, 1]; normalized internally to ImageNet stats.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.loss_network = nn.Sequential(*list(vgg.children())[:18]).eval().to(device)
        for p in self.loss_network.parameters():
            p.requires_grad = False

        self.criterion = nn.L1Loss()  # L1 in feature space is more robust than MSE
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inp    = (inp    - self.mean) / self.std
        target = (target - self.mean) / self.std
        return self.criterion(self.loss_network(inp), self.loss_network(target))


class FlowSmoothnessLoss(nn.Module):
    """
    Edge-aware flow smoothness loss (from XVFI / DAIN).
    Penalizes large flow gradients, down-weighted in textured regions
    where large gradients are expected (using image edge weights).

    loss = mean( exp(-|grad_img|) * |grad_flow| )
    """
    def forward(self, flow: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        # flow: (B, 2, H, W), img: (B, 3, H, W)
        # Image gradient magnitude (luma approximation)
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        grad_img_x = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        grad_img_y = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()

        # Flow gradients
        grad_flow_x = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
        grad_flow_y = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()

        # Edge-aware weights: suppress smoothness penalty at edges
        weight_x = torch.exp(-grad_img_x)  # (B, 1, H, W-1)
        weight_y = torch.exp(-grad_img_y)  # (B, 1, H-1, W)

        loss_x = (weight_x * grad_flow_x).mean()
        loss_y = (weight_y * grad_flow_y).mean()
        return loss_x + loss_y


class SSIMLoss(nn.Module):
    """1 - SSIM as a loss term. Uses a simple window-based implementation."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        kernel = self._gaussian_kernel(window_size, sigma)
        # (1, 1, ws, ws) — applied per channel
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return torch.outer(g, g)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        B, C, H, W = x.shape
        k = self.kernel.expand(C, 1, -1, -1)

        mu_x  = F.conv2d(x, k, padding=pad, groups=C)
        mu_y  = F.conv2d(y, k, padding=pad, groups=C)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sig_x  = F.conv2d(x * x, k, padding=pad, groups=C) - mu_x2
        sig_y  = F.conv2d(y * y, k, padding=pad, groups=C) - mu_y2
        sig_xy = F.conv2d(x * y, k, padding=pad, groups=C) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
        return (num / den).mean()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(x, y)


class TimeLensLoss(nn.Module):
    """L1 + lambda_perceptual * Perceptual (matching TimeLens paper)."""
    def __init__(self, device, lambda_perceptual=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perc = PerceptualLoss(device)
        self.lam_p = lambda_perceptual

    def forward(self, x, y, **kwargs):
        return self.l1(x, y) + self.lam_p * self.perc(x, y)


class EventWeightedCharbonnierLoss(nn.Module):
    """
    Charbonnier loss weighted by event density.
    Regions with dense events (= motion) receive higher loss weight,
    forcing the model to use event data rather than hedging with the mean frame.
    """
    def __init__(self, eps: float = 1e-3, alpha: float = 2.0):
        super().__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor, voxel: torch.Tensor) -> torch.Tensor:
        # voxel: (B, 5, H, W) — event voxel grid
        event_density = voxel.abs().sum(dim=1, keepdim=True)  # (B, 1, H, W)
        max_val = event_density.flatten(1).max(dim=1).values.view(-1, 1, 1, 1) + 1e-8
        weight = 1.0 + self.alpha * (event_density / max_val)  # [1, 1+alpha]

        diff = x - y
        per_pixel = torch.sqrt(diff * diff + self.eps * self.eps)
        return (weight * per_pixel).mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of:
      - Charbonnier reconstruction loss
      - VGG19 perceptual loss (L1 in feature space)
      - SSIM loss (1 - SSIM)
      - Edge-aware flow smoothness loss (applied to flow_t0 and flow_t1)
      - Event-weighted Charbonnier loss (upweights motion regions)

    During forward:
      x, y          — predicted and GT frames (B, 3, H, W)
      flow_t0/t1    — optional flow tensors for smoothness supervision
      f0            — optional reference frame (for edge-aware weighting)
      voxel         — optional event voxel for event-weighted loss
    """
    def __init__(
        self,
        device: torch.device,
        lambda_char: float       = 1.0,
        lambda_perceptual: float = 0.05,
        lambda_ssim: float       = 0.2,
        lambda_smooth: float     = 0.01,
        lambda_event_weighted: float = 0.0,
    ):
        super().__init__()
        self.char   = CharbonnierLoss()
        self.perc   = PerceptualLoss(device)
        self.ssim   = SSIMLoss()
        self.smooth = FlowSmoothnessLoss()
        self.evt_char = EventWeightedCharbonnierLoss()
        self.lam_c  = lambda_char
        self.lam_p  = lambda_perceptual
        self.lam_s  = lambda_ssim
        self.lam_sm = lambda_smooth
        self.lam_ew = lambda_event_weighted

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        flow_t0: torch.Tensor = None,
        flow_t1: torch.Tensor = None,
        f0: torch.Tensor      = None,
        voxel: torch.Tensor   = None,
    ) -> torch.Tensor:
        loss = self.lam_c * self.char(x, y)
        if self.lam_p > 0:
            loss = loss + self.lam_p * self.perc(x, y)
        if self.lam_s > 0:
            loss = loss + self.lam_s * self.ssim(x, y)

        if flow_t0 is not None and f0 is not None and self.lam_sm > 0:
            loss = loss + self.lam_sm * (
                self.smooth(flow_t0, f0) + self.smooth(flow_t1, f0)
            )

        if self.lam_ew > 0 and voxel is not None:
            loss = loss + self.lam_ew * self.evt_char(x, y, voxel)

        return loss
