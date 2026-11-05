"""Equivalent acceleration for official LIBERO-Plus glass and zoom noise.

Keep Gaussian operations, NumPy RandomState draws, and sequential pixel order.
In the reference tuple assignment both RHS rows are views: the second write
sees the first write. A conventional swap would change benchmark images.
"""
import numpy as np
from numba import njit
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom


@njit(cache=True)
def _shuffle_pixels(x, offsets, delta, iterations):
    n = 0
    for _ in range(iterations):
        for h in range(x.shape[0] - delta, delta, -1):
            for w in range(x.shape[1] - delta, delta, -1):
                dx, dy = offsets[n, 0], offsets[n, 1]
                n += 1
                # Exactly the reference's row-view assignment, not a swap.
                for k in range(x.shape[2]):
                    x[h, w, k] = x[h + dy, w + dx, k]


def glass_blur(x, severity=1):
    c = [
        (0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2),
        (1.1, 3, 2), (1.3, 3, 2), (1.5, 4, 2), (1.8, 4, 2),
        (2.2, 5, 1), (2.5, 5, 1),
    ][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)
    count = c[2] * max(0, x.shape[0] - 2 * c[1]) * max(0, x.shape[1] - 2 * c[1])
    # One contiguous draw has the same values and final RandomState as the
    # reference's consecutive size=(2,) calls; no RNG runs during pixel copies.
    offsets = np.random.randint(-c[1], c[1], size=(count, 2))
    _shuffle_pixels(x, offsets, c[1], c[2])
    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255


def clipped_zoom(img, zoom_factor):
    # The reference uses a 3D zoom with channel factor 1. Independent 2D
    # channel calls avoid interpolating the identity channel axis; keep the
    # same crop, output geometry, order=1 arithmetic, and float32 results.
    h = img.shape[0]
    ch = int(np.ceil(h / float(zoom_factor)))
    top = (h - ch) // 2
    crop = img[top:top + ch, top:top + ch]
    result = np.stack(
        [scizoom(crop[..., c], (zoom_factor, zoom_factor), order=1)
         for c in range(img.shape[2])], axis=-1
    )
    trim_top = (result.shape[0] - h) // 2
    return result[trim_top:trim_top + h, trim_top:trim_top + h]
