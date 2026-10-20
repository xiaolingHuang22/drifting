"""Shared display-space spectrogram image metrics (pixels normalized to [0,1])."""
import numpy as np
from scipy.ndimage import gaussian_filter


def image_metrics(reference, prediction):
    # Images in [0,1]. These are display-space metrics, not physical SEEG power.
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError(f"Expected equal RGB image shapes, got {reference.shape} and {prediction.shape}")
    if min(reference.shape[:2]) <= 10:
        raise ValueError("Images must be larger than ten pixels for the SSIM border crop")
    if not (np.isfinite(reference).all() and np.isfinite(prediction).all()):
        raise ValueError("Image metrics require finite pixels")
    x, y = reference.mean(axis=-1), prediction.mean(axis=-1)
    diff = prediction - reference
    mse = float(np.mean(diff ** 2))
    mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
    vx = np.maximum(gaussian_filter(x*x, 1.5) - mx*mx, 0)
    vy = np.maximum(gaussian_filter(y*y, 1.5) - my*my, 0)
    cov = gaussian_filter(x*y, 1.5) - mx*my
    ssim = ((2*mx*my + .01**2)*(2*cov + .03**2) /
            ((mx*mx+my*my+.01**2)*(vx+vy+.03**2)))
    return {'image_mse': mse, 'image_mae': float(np.abs(diff).mean()), 'image_rmse': mse**.5,
            'psnr_db': float(-10*np.log10(max(mse, 1e-12))),
            'ssim_gray': float(ssim[5:-5, 5:-5].mean()),
            'row_profile_mae': float(np.abs(x.mean(1)-y.mean(1)).mean()),
            'column_profile_mae': float(np.abs(x.mean(0)-y.mean(0)).mean()),
            'gradient_mae': float(np.mean([np.abs(np.diff(x, axis=a)-np.diff(y, axis=a)).mean() for a in (0, 1)]))}

