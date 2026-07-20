"""Image-quality metrics and a guided denoiser, all in pure numpy.

* :func:`psnr` and :func:`ssim` score a render against a high-sample reference.
* :func:`bilateral_denoise` is a normal/depth-guided edge-preserving filter --
  the minimal idea behind production denoisers (OptiX, DLSS): trade Monte-Carlo
  samples for a cheap spatial filter that respects geometric edges.
"""
from __future__ import annotations

import numpy as np

MAX_PSNR = 100.0     # cap for identical images (avoid infinity)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return MAX_PSNR
    return min(MAX_PSNR, 10.0 * np.log10(1.0 / mse))


def _ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
    """Global SSIM on a single channel (means/vars over the whole image)."""
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - mx) * (y - my)).mean()
    num = (2 * mx * my + c1) * (2 * cxy + c2)
    den = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(num / den)


def ssim(a: np.ndarray, b: np.ndarray, window: int = 7) -> float:
    """Mean windowed SSIM over the image, averaged across colour channels."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 2:
        a = a[..., None]
        b = b[..., None]
    h, w, c = a.shape
    win = min(window, h, w)
    step = max(1, win // 2)
    scores = []
    for ch in range(c):
        for i in range(0, h - win + 1, step):
            for j in range(0, w - win + 1, step):
                scores.append(_ssim_gray(a[i:i + win, j:j + win, ch],
                                         b[i:i + win, j:j + win, ch]))
    return float(np.mean(scores)) if scores else _ssim_gray(a[..., 0], b[..., 0])


def bilateral_denoise(img: np.ndarray, gbuf: dict, sigma_spatial: float = 1.5,
                      sigma_color: float = 0.12, radius: int = 2) -> np.ndarray:
    """Cross-bilateral filter guided by G-buffer normals and depth.

    Weights combine spatial distance, colour similarity, and geometric
    similarity (normal + depth), so the filter smooths Monte-Carlo noise inside
    a surface without blurring across silhouettes.
    """
    h, w, _ = img.shape
    nrm = gbuf["normal"]
    dep = gbuf["depth"]
    out = np.zeros_like(img)
    wsum = np.zeros((h, w, 1))
    inv_s = 1.0 / (2 * sigma_spatial ** 2)
    inv_c = 1.0 / (2 * sigma_color ** 2)
    depth_scale = 1.0 / (np.median(dep[dep < 1e3]) + 1e-6) if np.any(dep < 1e3) else 1.0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            s_nrm = np.roll(np.roll(nrm, dy, axis=0), dx, axis=1)
            s_dep = np.roll(np.roll(dep, dy, axis=0), dx, axis=1)
            wsp = np.exp(-(dx * dx + dy * dy) * inv_s)
            dcol = np.sum((img - shifted) ** 2, axis=2, keepdims=True)
            wcol = np.exp(-dcol * inv_c)
            ndot = np.clip(np.sum(nrm * s_nrm, axis=2, keepdims=True), 0, 1)
            wnrm = ndot ** 8
            ddep = ((dep - s_dep) * depth_scale)[..., None]
            wdep = np.exp(-(ddep ** 2) * 4.0)
            weight = wsp * wcol * wnrm * wdep
            out += weight * shifted
            wsum += weight
    return out / np.maximum(wsum, 1e-8)
