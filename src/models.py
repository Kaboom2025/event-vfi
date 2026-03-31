import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _groupnorm(num_channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 8 groups, compatible with small channel counts."""
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class DoubleConv(nn.Module):
    """(Conv => GroupNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _groupnorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _groupnorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Legacy models (kept for ablation / checkpoint compatibility)
# ---------------------------------------------------------------------------

class RGBBaselineUNet(nn.Module):
    """Ablation baseline: RGB-only (no events), 6-channel input."""
    def __init__(self, n_channels=6, n_classes=3, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512,  256 // factor, bilinear)
        self.up3   = Up(256,  128 // factor, bilinear)
        self.up4   = Up(128,  64,            bilinear)
        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        mean_frame = (x[:, :3] + x[:, 3:]) / 2
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        residual = self.outc(x)
        return torch.clamp(mean_frame + residual, 0.0, 1.0)


def _fuse_conv(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


class DualEncoderUNet(nn.Module):
    """Dual-encoder U-Net (direct synthesis). Kept for ablation."""
    def __init__(self, n_channels_rgb=6, n_channels_evt=5, n_classes=3, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc_rgb   = DoubleConv(n_channels_rgb, 64)
        self.down1_rgb = Down(64,  128)
        self.down2_rgb = Down(128, 256)
        self.down3_rgb = Down(256, 512)
        self.down4_rgb = Down(512, 1024 // factor)

        self.inc_evt   = DoubleConv(n_channels_evt, 64)
        self.down1_evt = Down(64,  128)
        self.down2_evt = Down(128, 256)
        self.down3_evt = Down(256, 512)
        self.down4_evt = Down(512, 1024 // factor)

        c5 = 1024 // factor
        self.fuse5 = _fuse_conv(c5  * 2, c5)
        self.fuse4 = _fuse_conv(512 * 2, 512)
        self.fuse3 = _fuse_conv(256 * 2, 256)
        self.fuse2 = _fuse_conv(128 * 2, 128)
        self.fuse1 = _fuse_conv(64  * 2, 64)

        self.up1  = Up(1024, 512 // factor, bilinear)
        self.up2  = Up(512,  256 // factor, bilinear)
        self.up3  = Up(256,  128 // factor, bilinear)
        self.up4  = Up(128,  64,            bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, rgb, evt):
        mean_frame = (rgb[:, :3] + rgb[:, 3:]) / 2
        r1 = self.inc_rgb(rgb);   e1 = self.inc_evt(evt)
        r2 = self.down1_rgb(r1);  e2 = self.down1_evt(e1)
        r3 = self.down2_rgb(r2);  e3 = self.down2_evt(e2)
        r4 = self.down3_rgb(r3);  e4 = self.down3_evt(e3)
        r5 = self.down4_rgb(r4);  e5 = self.down4_evt(e4)

        x5 = self.fuse5(torch.cat([r5, e5], dim=1))
        s4 = self.fuse4(torch.cat([r4, e4], dim=1))
        s3 = self.fuse3(torch.cat([r3, e3], dim=1))
        s2 = self.fuse2(torch.cat([r2, e2], dim=1))
        s1 = self.fuse1(torch.cat([r1, e1], dim=1))

        x = self.up1(x5, s4)
        x = self.up2(x,  s3)
        x = self.up3(x,  s2)
        x = self.up4(x,  s1)
        residual = self.outc(x)
        return torch.clamp(mean_frame + residual, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Warp utility
# ---------------------------------------------------------------------------

def backwarp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Backward-warp `img` using `flow` (pixel displacement).
    img:  (B, C, H, W)
    flow: (B, 2, H, W)  — flow[:, 0] = x-displacement, flow[:, 1] = y-displacement
    Returns warped image (B, C, H, W).
    """
    B, _, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing='ij',
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1) + flow[:, 0]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1) + flow[:, 1]

    # Normalize to [-1, 1] for grid_sample
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)


# ---------------------------------------------------------------------------
# Flow-based architecture: EventWarpNet (improved)
# ---------------------------------------------------------------------------

class FlowNet(nn.Module):
    """
    U-Net that estimates bidirectional flows from events + RGB.
    Input:  cat(f0, f1, events) = 11 channels
    Output: 4 channels — flow_t0 (2ch) and flow_t1 (2ch)

    Wider than before (64 base channels) to increase capacity.
    """
    def __init__(self, in_channels=11, base_ch=64, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1
        c = base_ch

        self.inc   = DoubleConv(in_channels, c)
        self.down1 = Down(c,     c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 8 // factor)

        self.up1 = Up(c * 8 + c * 8 // factor, c * 4 // factor, bilinear)
        self.up2 = Up(c * 4 + c * 4 // factor, c * 2 // factor, bilinear)
        self.up3 = Up(c * 2 + c * 2 // factor, c,               bilinear)
        self.up4 = Up(c + c,                   c,               bilinear)
        self.outc = nn.Conv2d(c, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)


class ContextEncoderV1(nn.Module):
    """
    Legacy context encoder (v1) — matches vimeo_v7_warp checkpoint layout.
    Conv2d(3,ctx_ch,7,bias=False) → GroupNorm → LeakyReLU →
    Conv2d(ctx_ch,ctx_ch,3,bias=False) → GroupNorm → LeakyReLU
    """
    def __init__(self, ctx_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ctx_ch, 7, padding=3, bias=False),
            _groupnorm(ctx_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ctx_ch, ctx_ch, 3, padding=1, bias=False),
            _groupnorm(ctx_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ContextEncoder(nn.Module):
    """
    Extracts context features from a single RGB frame.
    Warped context features help RefineNet reconstruct detail that
    bilinear warping destroys. Deeper than v1 (4 layers, 64ch output).
    Input:  (B, 3, H, W)
    Output: (B, ctx_ch, H, W)
    """
    def __init__(self, ctx_ch=64):
        super().__init__()
        mid = max(32, ctx_ch // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, mid, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid, ctx_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ctx_ch, ctx_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RefineNet(nn.Module):
    """
    Deeper U-Net that predicts per-frame occlusion masks + residual.

    Input channels = warped_f0(3) + warped_f1(3) + events(5) + flow_t0(2) + flow_t1(2)
                   + ctx_f0(ctx_ch) + ctx_f1(ctx_ch) = 15 + 2*ctx_ch

    Output: mask0 (1ch, sigmoid) + mask1 (1ch, sigmoid) + residual (3ch)
      - mask0: how much to trust warped_f0 at each pixel
      - mask1: how much to trust warped_f1 at each pixel
      - residual: additive correction for occluded/incorrect regions
    """
    def __init__(self, ctx_ch=32, base_ch=64, bilinear=True):
        super().__init__()
        in_channels = 15 + 2 * ctx_ch
        factor = 2 if bilinear else 1
        c = base_ch

        self.inc   = DoubleConv(in_channels, c)
        self.down1 = Down(c,     c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 4 // factor)

        self.up1 = Up(c * 4 + c * 4 // factor, c * 2 // factor, bilinear)
        self.up2 = Up(c * 2 + c * 2 // factor, c,               bilinear)
        self.up3 = Up(c + c,                   c,               bilinear)
        # 4 outputs: alpha (1) + residual (3)
        self.outc = nn.Conv2d(c, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x,  x2)
        x = self.up3(x,  x1)
        out = self.outc(x)
        alpha    = torch.sigmoid(out[:, :1])
        residual = torch.tanh(out[:, 1:]) * 0.2
        return alpha, residual


class EventWarpNet(nn.Module):
    """
    Improved flow-based event-guided video frame interpolation.

    Changes vs. v1:
    - Wider FlowNet (64 base channels)
    - Context encoder: warped context features fed to RefineNet (DAIN-inspired)
    - Single alpha mask for blending warped frames
    - Bounded residual (tanh * 0.2) for stable gradients

    Forward returns (pred, flow_t0, flow_t1) during training so the caller can
    compute flow-smoothness loss. At inference call model.infer(rgb, evt).
    """
    def __init__(self, ctx_ch=32):
        super().__init__()
        self.ctx_ch     = ctx_ch
        self.flownet    = FlowNet(in_channels=11, base_ch=64)
        self.ctx_enc    = ContextEncoderV1(ctx_ch=ctx_ch)
        self.refinenet  = RefineNet(ctx_ch=ctx_ch, base_ch=64)

    def forward(self, rgb: torch.Tensor, evt: torch.Tensor):
        """
        rgb: (B, 6, H, W) = cat(f0, f1)
        evt: (B, 5, H, W)
        Returns: (pred, flow_t0, flow_t1)
          pred:     (B, 3, H, W) interpolated frame in [0, 1]
          flow_t0:  (B, 2, H, W) flow from target t to frame 0
          flow_t1:  (B, 2, H, W) flow from target t to frame 1
        """
        f0 = rgb[:, :3]
        f1 = rgb[:, 3:]

        # Step 1: estimate bidirectional flow from events + RGB
        flows   = self.flownet(torch.cat([f0, f1, evt], dim=1))
        flow_t0 = flows[:, :2]
        flow_t1 = flows[:, 2:]

        # Step 2: warp input frames to target time
        warped_f0 = backwarp(f0, flow_t0)
        warped_f1 = backwarp(f1, flow_t1)

        # Step 3: context features — encode then warp alongside frames
        ctx_f0 = backwarp(self.ctx_enc(f0), flow_t0)
        ctx_f1 = backwarp(self.ctx_enc(f1), flow_t1)

        # Step 4: refine — predict alpha mask + bounded residual
        refine_in = torch.cat([warped_f0, warped_f1, evt, flow_t0, flow_t1,
                                ctx_f0, ctx_f1], dim=1)
        alpha, residual = self.refinenet(refine_in)

        # Step 5: blend and add residual
        out = alpha * warped_f0 + (1 - alpha) * warped_f1 + residual
        pred = torch.clamp(out, 0.0, 1.0)

        return pred, flow_t0, flow_t1

    def infer(self, rgb: torch.Tensor, evt: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for inference — returns prediction only."""
        pred, _, _ = self.forward(rgb, evt)
        return pred


# ---------------------------------------------------------------------------
# EventAttentionGate — spatial attention from events for RefineNet
# ---------------------------------------------------------------------------

class EventAttentionGate(nn.Module):
    """Events generate spatial attention to focus refinement on motion regions."""
    def __init__(self, evt_ch=5, feat_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(evt_ch, feat_ch, 3, padding=1, bias=False)
        self.gn    = _groupnorm(feat_ch)
        self.conv2 = nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=True)  # bias for init control
        # Initialize bias so sigmoid starts near 1.0 (identity), then learns to suppress
        nn.init.constant_(self.conv2.bias, 1.5)

    def forward(self, features, events):
        attn = torch.sigmoid(self.conv2(F.relu(self.gn(self.conv1(events)))))
        return features * attn


class AttentiveRefineNet(nn.Module):
    """
    RefineNet with event attention gates at each decoder level.
    Events are downsampled to match each scale and used to modulate features.
    """
    def __init__(self, ctx_ch=32, base_ch=64, evt_ch=5, bilinear=True):
        super().__init__()
        in_channels = 15 + 2 * ctx_ch
        factor = 2 if bilinear else 1
        c = base_ch

        self.inc   = DoubleConv(in_channels, c)
        self.down1 = Down(c,     c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 4 // factor)

        # Event attention gates at each decoder level
        self.attn3 = EventAttentionGate(evt_ch, c * 4 // factor)
        self.attn2 = EventAttentionGate(evt_ch, c * 2 // factor)
        self.attn1 = EventAttentionGate(evt_ch, c)

        self.up1 = Up(c * 4 + c * 4 // factor, c * 2 // factor, bilinear)
        self.up2 = Up(c * 2 + c * 2 // factor, c,               bilinear)
        self.up3 = Up(c + c,                   c,               bilinear)
        self.outc = nn.Conv2d(c, 4, kernel_size=3, padding=1)

    def forward(self, x, events):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Apply attention at bottleneck, pooling events to match spatial size
        x4 = self.attn3(x4, F.adaptive_avg_pool2d(events, x4.shape[2:]))
        x = self.up1(x4, x3)
        x = self.attn2(x, F.adaptive_avg_pool2d(events, x.shape[2:]))
        x = self.up2(x, x2)
        x = self.attn1(x, F.adaptive_avg_pool2d(events, x.shape[2:]))
        x = self.up3(x, x1)

        out = self.outc(x)
        alpha    = torch.sigmoid(out[:, :1])
        residual = out[:, 1:]  # unbounded — final clamp handles range
        return alpha, residual


# ---------------------------------------------------------------------------
# Two-Stage Flow: events first, RGB refines (TimeLens-inspired)
# ---------------------------------------------------------------------------

class TwoStageFlowNet(nn.Module):
    """
    Stage 1: Events-only → coarse bidirectional flow.
    Stage 2: RGB + coarse warp + coarse flow → residual flow refinement.
    Returns (refined_flow_t0, refined_flow_t1, coarse_flow_t0, coarse_flow_t1).
    """
    def __init__(self, evt_ch=5, base_ch=48):
        super().__init__()
        self.event_flow  = FlowNet(in_channels=evt_ch, base_ch=base_ch)
        self.flow_refine = FlowNet(in_channels=13, base_ch=base_ch)  # f0(3)+f1(3)+coarse_warp(3)+coarse_flow(4)

    def forward(self, f0, f1, evt):
        # Stage 1: coarse flow from events alone
        coarse_flow = self.event_flow(evt)
        flow_t0_c = coarse_flow[:, :2]
        flow_t1_c = coarse_flow[:, 2:]

        # Warp with coarse flow for refinement context
        coarse_warp = 0.5 * backwarp(f0, flow_t0_c) + 0.5 * backwarp(f1, flow_t1_c)

        # Stage 2: refine flow using RGB appearance
        refine_in = torch.cat([f0, f1, coarse_warp, coarse_flow], dim=1)
        delta_flow = self.flow_refine(refine_in)

        refined_flow = coarse_flow + delta_flow
        return refined_flow[:, :2], refined_flow[:, 2:], flow_t0_c, flow_t1_c


class EventWarpNetV2(nn.Module):
    """
    Improved EventWarpNet with:
    1. Two-stage flow: events estimate coarse flow, RGB refines it
    2. Event attention gates in RefineNet: focus on motion regions

    Forward returns (pred, flow_t0, flow_t1) for loss computation.
    """
    def __init__(self, ctx_ch=64):
        super().__init__()
        self.ctx_ch    = ctx_ch
        self.flownet   = TwoStageFlowNet(evt_ch=5, base_ch=48)
        self.ctx_enc   = ContextEncoder(ctx_ch=ctx_ch)
        self.refinenet = AttentiveRefineNet(ctx_ch=ctx_ch, base_ch=64, evt_ch=5)

    def forward(self, rgb: torch.Tensor, evt: torch.Tensor):
        f0 = rgb[:, :3]
        f1 = rgb[:, 3:]

        # Step 1: two-stage flow estimation (returns refined + coarse flows)
        flow_t0, flow_t1, flow_t0_c, flow_t1_c = self.flownet(f0, f1, evt)

        # Step 2: warp frames with refined flow
        warped_f0 = backwarp(f0, flow_t0)
        warped_f1 = backwarp(f1, flow_t1)

        # Step 3: warp context features
        ctx_f0 = backwarp(self.ctx_enc(f0), flow_t0)
        ctx_f1 = backwarp(self.ctx_enc(f1), flow_t1)

        # Step 4: attentive refinement (events modulate decoder features)
        refine_in = torch.cat([warped_f0, warped_f1, evt, flow_t0, flow_t1,
                               ctx_f0, ctx_f1], dim=1)
        alpha, residual = self.refinenet(refine_in, evt)

        # Step 5: blend
        out = alpha * warped_f0 + (1 - alpha) * warped_f1 + residual
        pred = torch.clamp(out, 0.0, 1.0)

        return pred, flow_t0, flow_t1, flow_t0_c, flow_t1_c

    def infer(self, rgb: torch.Tensor, evt: torch.Tensor) -> torch.Tensor:
        pred, _, _, _, _ = self.forward(rgb, evt)
        return pred


# ---------------------------------------------------------------------------
# SizeAdapter (from TimeLens)
# ---------------------------------------------------------------------------

import math

class SizeAdapter:
    """Pads input to multiple of minimum_size, then crops output back."""

    def __init__(self, minimum_size=32):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def pad(self, x):
        h, w = x.size()[-2:]
        self._pixels_pad_to_height = int(math.ceil(h / self._minimum_size) * self._minimum_size) - h
        self._pixels_pad_to_width = int(math.ceil(w / self._minimum_size) * self._minimum_size) - w
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(x)

    def unpad(self, x):
        return x[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]


# ---------------------------------------------------------------------------
# TimeLens synthesis-only U-Net (Section 3.2, Fig 3b)
# ---------------------------------------------------------------------------

class _TLDown(nn.Module):
    """TimeLens down block: AvgPool(2) + Conv + LeakyReLU + Conv + LeakyReLU."""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class _TLUp(nn.Module):
    """TimeLens up block: bilinear 2x + conv + cat(skip) + conv, LeakyReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(2 * out_ch, out_ch, 3, padding=1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(torch.cat([x, skip], dim=1)), negative_slope=0.1)
        return x


class SynthesisUNet(nn.Module):
    """
    TimeLens synthesis-only module (SuperSloMo UNet variant).

    Input:  cat(f0, voxel, f1) = 11 channels  [3 + 5 + 3]
    Output: 3 channels with sigmoid — directly regresses middle frame.

    Architecture faithfully replicates timelens/superslomo/unet.py:
      - LeakyReLU(0.1) everywhere, no GroupNorm, no residual learning
      - AvgPool for downsampling, bilinear interpolation for upsampling
      - SizeAdapter pads to multiple of 32
    """

    def __init__(self, in_channels=11):
        super().__init__()
        self._size_adapter = SizeAdapter(minimum_size=32)
        self.conv1 = nn.Conv2d(in_channels, 32, 7, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, padding=3)
        self.down1 = _TLDown(32, 64, 5)
        self.down2 = _TLDown(64, 128, 3)
        self.down3 = _TLDown(128, 256, 3)
        self.down4 = _TLDown(256, 512, 3)
        self.down5 = _TLDown(512, 512, 3)
        self.up1 = _TLUp(512, 512)
        self.up2 = _TLUp(512, 256)
        self.up3 = _TLUp(256, 128)
        self.up4 = _TLUp(128, 64)
        self.up5 = _TLUp(64, 32)
        self.conv3 = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        x = self._size_adapter.pad(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = self.conv3(x)
        x = self._size_adapter.unpad(x)
        return x
