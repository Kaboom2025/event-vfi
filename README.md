# Event-Guided Video Frame Interpolation

**UW CSE 493 — Neuromorphic Vision**

Neural video frame interpolation using synthetic event camera data. We fuse low-res RGB frames with simulated event voxel streams to reconstruct sharp intermediate frames — achieving **34.09 dB PSNR / 0.945 SSIM**, a +4.14 dB improvement over RGB-only baselines. Four architectures were built and compared, culminating in a two-stage flow estimation pipeline with learned attention gates.

## The Problem

Standard cameras capture frames at fixed intervals (30 fps). Between those frames, the world keeps moving — fast-moving objects blur, ghost, or disappear entirely when you try to reconstruct intermediate frames. Event cameras solve this by firing at every pixel that changes brightness, with microsecond resolution. We use **v2e** to simulate event camera data from standard video, then fuse these events with RGB frames to guide interpolation.

## Our Approach

A two-stage pipeline:

1. **Event Flow Stage**: A FlowNet takes the 5-bin event voxel grid and estimates coarse bidirectional optical flow — capturing fast motion that RGB alone misses.
2. **RGB Refinement Stage**: A second network refines the coarse flow using RGB appearance, predicting a delta flow for texture-level details.
3. **Attention Gates**: Learned spatial attention focuses refinement on high-motion regions — discovered from the reconstruction loss alone, with zero explicit supervision.

The final frame is produced by backward-warping both boundary frames using the estimated flow, then blending with a learned alpha mask and residual correction.

## Results

| Model | PSNR (dB) | SSIM | Params |
|-------|-----------|------|--------|
| RGB Baseline | 29.95 | 0.8708 | 17.3M |
| DualEncoder (zero events) | 28.00 | 0.8296 | 27.9M |
| DualEncoder | 31.00 | 0.8860 | 27.9M |
| DualEncoder + Perceptual | 31.24 | 0.8910 | 27.9M |
| EventWarpNet v1 | 32.73 | 0.9219 | 13.4M |
| EventWarpNetV2 | 33.83 | 0.9408 | 15.1M |
| **EventWarpNetV2 + Perceptual** | **34.09** | **0.9450** | **15.1M** |

## Key Findings

- Events provide **+3.0 dB** genuine signal (zero-events ablation: 28.00 vs 31.00 dB)
- Flow-based warping beats direct synthesis by **+2.85 dB**
- Two-stage decomposition (events to coarse flow, RGB to refined flow) is optimal
- Learned attention gates focus refinement on motion regions without explicit supervision
- The benefit of events concentrates on high-motion scenes

## Tech Stack

Python, PyTorch, v2e (event simulator), Google Colab, React, Tailwind CSS

## Dataset

Vimeo-90k triplet test set (3,782 triplets, 448x256). Events simulated via v2e with default parameters. 5-bin temporal voxel grid representation.
