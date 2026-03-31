import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def calculate_psnr(pred, gt):
    """
    Calculate PSNR for a batch of images.
    Expects tensors in [0, 1] range.
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    return psnr_metric(pred, gt)

def calculate_ssim(pred, gt):
    """
    Calculate SSIM for a batch of images.
    Expects tensors in [0, 1] range.
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    return ssim_metric(pred, gt)
