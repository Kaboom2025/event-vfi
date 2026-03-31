import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def load_h5_events(h5_path):
    with h5py.File(h5_path, 'r') as f:
        if 'events' in f:
            return f['events'][:]
        print(f"Keys in h5: {list(f.keys())}")
        return None


def events_to_voxel_grid(events, num_bins, width, height, device='cpu'):
    voxel = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)
    if events.shape[0] == 0:
        return voxel
    ts = torch.from_numpy(events[:, 0]).float().to(device)
    xs = torch.from_numpy(events[:, 1]).long().to(device)
    ys = torch.from_numpy(events[:, 2]).long().to(device)
    ps = torch.from_numpy(events[:, 3]).float().to(device)
    if ts[-1] != ts[0]:
        ts = (ts - ts[0]) / (ts[-1] - ts[0]) * (num_bins - 1)
    else:
        ts = torch.zeros_like(ts)
    tis = ts.long()
    dts = ts - tis.float()
    vals_left  = ps * (1.0 - dts)
    vals_right = ps * dts
    voxel = voxel.flatten()
    valid = tis < num_bins
    idx_l = xs[valid] + ys[valid] * width + tis[valid] * width * height
    voxel.index_add_(0, idx_l, vals_left[valid])
    valid_r = valid & ((tis + 1) < num_bins)
    idx_r = xs[valid_r] + ys[valid_r] * width + (tis[valid_r] + 1) * width * height
    voxel.index_add_(0, idx_r, vals_right[valid_r])
    return voxel.view(num_bins, height, width)


def normalize_voxel(voxel):
    mask = voxel != 0
    if mask.any():
        mean = voxel[mask].mean()
        std  = voxel[mask].std()
        voxel[mask] = (voxel[mask] - mean) / (std + 1e-8)
    return voxel

class EventFrameDataset(Dataset):
    def __init__(self, root_dir, event_root, list_file, patch_size=256, augment=True):
        """
        Args:
            root_dir: Base directory for RGB frames (e.g. data/X4K1000FPS)
            event_root: Base directory for voxels (e.g. data/X4K1000FPS_events)
            list_file: Path to text file containing relative sample paths
            patch_size: Size of random crops
            augment: Whether to apply data augmentation
        """
        with open(list_file, 'r') as f:
            self.samples = [line.strip() for line in f if line.strip()]

        self.root = root_dir
        self.event_root = event_root
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_rel_path = self.samples[idx]

        # Load frames (0, 16, 32 for a typical triplet in our subset)
        # Using hardcoded names for now based on X4K decoding
        sample_path = os.path.join(self.root, sample_rel_path)
        f0 = Image.open(os.path.join(sample_path, "0000.png")).convert('RGB')
        gt = Image.open(os.path.join(sample_path, "0016.png")).convert('RGB')
        f1 = Image.open(os.path.join(sample_path, "0032.png")).convert('RGB')

        # Load voxel
        voxel_path = os.path.join(self.event_root, sample_rel_path, "voxel.pt")
        voxel = torch.load(voxel_path, weights_only=True)  # (B, H, W)

        # Random crop
        w, h = f0.size
        if self.augment or (w > self.patch_size or h > self.patch_size):
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
            box = (x, y, x + self.patch_size, y + self.patch_size)

            f0 = f0.crop(box)
            gt = gt.crop(box)
            f1 = f1.crop(box)
            voxel = voxel[:, y : y + self.patch_size, x : x + self.patch_size]

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                # Horizontal flip
                f0 = f0.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
                f1 = f1.transpose(Image.FLIP_LEFT_RIGHT)
                voxel = voxel.flip(-1)

            if random.random() > 0.5:
                # Temporal reversal
                f0, f1 = f1, f0
                # Voxel should also be reversed temporally (approx)
                voxel = voxel.flip(0)

        return self.to_tensor(f0), self.to_tensor(f1), voxel, self.to_tensor(gt)


class ProcessedDataset(Dataset):
    """
    Loads pre-processed clips. Each clip_dir contains:
      f0_blur.png  — blurred/downscaled first frame
      f1_blur.png  — blurred/downscaled last frame
      gt.png       — ground-truth middle frame
      voxel.pt     — 5×H×W event voxel tensor
    """
    def __init__(self, clip_dirs, patch_size=256, augment=True, event_gain=1.0, zero_events=False):
        self.clip_dirs   = clip_dirs
        self.patch_size  = patch_size
        self.augment     = augment
        self.event_gain  = event_gain
        self.zero_events = zero_events
        self.to_tensor   = T.ToTensor()

    def __len__(self):
        return len(self.clip_dirs)

    def __getitem__(self, idx):
        d = self.clip_dirs[idx]

        f0    = Image.open(os.path.join(d, 'f0_blur.png')).convert('RGB')
        f1    = Image.open(os.path.join(d, 'f1_blur.png')).convert('RGB')
        gt    = Image.open(os.path.join(d, 'gt.png')).convert('RGB')
        voxel = torch.load(os.path.join(d, 'voxel.pt'), weights_only=True)
        # NOTE: voxel is already normalized during data prep (create_voxel_grid).
        # Do NOT call normalize_voxel again — double normalization distorts the signal.

        # Zero-events ablation: controls for model capacity vs event utility
        if self.zero_events:
            voxel = torch.zeros_like(voxel)

        # Align voxel to RGB: center-crop voxel to match RGB aspect ratio, then resize.
        # Voxel is 768x768 (square), RGB may be 640x360 (16:9).
        # Squashing would break spatial correspondence — crop instead.
        w, h = f0.size   # PIL: (width, height)
        vH, vW = voxel.shape[1], voxel.shape[2]
        if vH != h or vW != w:
            rgb_aspect = w / h
            voxel_aspect = vW / vH
            if voxel_aspect > rgb_aspect:
                # Voxel is wider — crop width
                new_vW = int(vH * rgb_aspect)
                left = (vW - new_vW) // 2
                voxel = voxel[:, :, left : left + new_vW]
            elif voxel_aspect < rgb_aspect:
                # Voxel is taller — crop height
                new_vH = int(vW / rgb_aspect)
                top = (vH - new_vH) // 2
                voxel = voxel[:, top : top + new_vH, :]
            voxel = F.interpolate(
                voxel.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0)

        # Boost event signal
        if self.event_gain != 1.0:
            voxel = voxel * self.event_gain

        # Random crop (applied consistently across all modalities)
        ps = self.patch_size
        x = random.randint(0, max(0, w - ps))
        y = random.randint(0, max(0, h - ps))
        box = (x, y, x + ps, y + ps)

        f0    = f0.crop(box)
        f1    = f1.crop(box)
        gt    = gt.crop(box)
        voxel = voxel[:, y : y + ps, x : x + ps]

        if self.augment:
            # Horizontal flip (50%)
            if random.random() > 0.5:
                f0    = f0.transpose(Image.FLIP_LEFT_RIGHT)
                f1    = f1.transpose(Image.FLIP_LEFT_RIGHT)
                gt    = gt.transpose(Image.FLIP_LEFT_RIGHT)
                voxel = voxel.flip(-1)

            # Vertical flip (50%)
            if random.random() > 0.5:
                f0    = f0.transpose(Image.FLIP_TOP_BOTTOM)
                f1    = f1.transpose(Image.FLIP_TOP_BOTTOM)
                gt    = gt.transpose(Image.FLIP_TOP_BOTTOM)
                voxel = voxel.flip(-2)

            # Temporal reversal (50%)
            if random.random() > 0.5:
                f0, f1 = f1, f0
                voxel  = voxel.flip(0)

        return self.to_tensor(f0), self.to_tensor(f1), voxel, self.to_tensor(gt)


class VimeoTripletDataset(Dataset):
    """
    Loads Vimeo-90k style triplets. Each triplet directory contains:
      im1.png   — first frame (f0)
      im3.png   — last frame (f1)
      im2.png   — ground-truth middle frame
      voxel.pt  — 5×H×W event voxel tensor (generated at native resolution)

    Args:
        triplet_dirs: list of absolute paths to triplet directories
        patch_size: random crop size (default 256)
        augment: whether to apply augmentation (flip, temporal reversal)
        event_gain: multiplicative scaling for event voxel (default 1.0)
    """
    def __init__(self, triplet_dirs, patch_size=256, augment=True, event_gain=1.0, zero_events=False):
        self.triplet_dirs  = triplet_dirs
        self.patch_size    = patch_size
        self.augment       = augment
        self.event_gain    = event_gain
        self.zero_events   = zero_events
        self.to_tensor     = T.ToTensor()

    def __len__(self):
        return len(self.triplet_dirs)

    def __getitem__(self, idx):
        d = self.triplet_dirs[idx]

        f0    = Image.open(os.path.join(d, 'im1.png')).convert('RGB')
        f1    = Image.open(os.path.join(d, 'im3.png')).convert('RGB')
        gt    = Image.open(os.path.join(d, 'im2.png')).convert('RGB')
        voxel = torch.load(os.path.join(d, 'voxel.pt'), weights_only=True)
        # NOTE: voxel is already normalized during data prep. No double normalization.

        # Zero-events ablation
        if self.zero_events:
            voxel = torch.zeros_like(voxel)

        # Event gain
        if self.event_gain != 1.0:
            voxel = voxel * self.event_gain

        # Random crop
        w, h = f0.size  # PIL: (width, height)
        ps = self.patch_size
        if w >= ps and h >= ps:
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            box = (x, y, x + ps, y + ps)
            f0    = f0.crop(box)
            f1    = f1.crop(box)
            gt    = gt.crop(box)
            voxel = voxel[:, y : y + ps, x : x + ps]

        if self.augment:
            # Horizontal flip (50%)
            if random.random() > 0.5:
                f0    = f0.transpose(Image.FLIP_LEFT_RIGHT)
                f1    = f1.transpose(Image.FLIP_LEFT_RIGHT)
                gt    = gt.transpose(Image.FLIP_LEFT_RIGHT)
                voxel = voxel.flip(-1)

            # Vertical flip (50%)
            if random.random() > 0.5:
                f0    = f0.transpose(Image.FLIP_TOP_BOTTOM)
                f1    = f1.transpose(Image.FLIP_TOP_BOTTOM)
                gt    = gt.transpose(Image.FLIP_TOP_BOTTOM)
                voxel = voxel.flip(-2)

            # Temporal reversal (50%)
            if random.random() > 0.5:
                f0, f1 = f1, f0
                voxel  = voxel.flip(0)

        return self.to_tensor(f0), self.to_tensor(f1), voxel, self.to_tensor(gt)
