"""
Dijagnostika: koja maska iz thresholdinga pravi sum u birds-eye prikazu.

Pokretanje (iz root foldera repoa):
    python -m src.debug_masks test1.jpg test4.jpg

Za svaku sliku pravi output_images/debug_masks_<ime>.png sa 6 panela
(sve vec u birds-eye prikazu):
    boja | gradijent | saturacija
    zuta | bela      | kombinovano
"""
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.calibration import ROOT, load_calibration, undistort
from src.perspective import warp
from src import tresholding as th

TEST_DIR = ROOT / "test_images"
OUT_DIR = ROOT / "output_images"


def masks(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_ch, s_ch = hls[:, :, 1], hls[:, :, 2]
    grad = th.sobel_x_binary(l_ch)
    sat = th.channel_binary(s_ch, th.S_THRESH) & (l_ch >= th.S_MIN_LIGHTNESS)
    yellow = (cv2.inRange(hls, th.YELLOW_LOW, th.YELLOW_HIGH) > 0).astype(np.uint8)
    white = (cv2.inRange(hls, th.WHITE_LOW, th.WHITE_HIGH) > 0).astype(np.uint8)
    combined = th.threshold(img)
    return grad, sat, yellow, white, combined


def main(names):
    mtx, dist = load_calibration()
    for name in names:
        img = undistort(cv2.imread(str(TEST_DIR / name)), mtx, dist)
        grad, sat, yellow, white, combined = masks(img)

        panels = [
            (cv2.cvtColor(warp(img), cv2.COLOR_BGR2RGB), "Color (warped)"),
            (warp(grad), f"Gradient  ({warp(grad).sum()} px)"),
            (warp(sat), f"Saturation  ({warp(sat).sum()} px)"),
            (warp(yellow), f"Yellow  ({warp(yellow).sum()} px)"),
            (warp(white), f"White  ({warp(white).sum()} px)"),
            (warp(combined), "Combined"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 7), facecolor="#1e1e1e")
        for ax, (data, title) in zip(axes.ravel(), panels):
            ax.imshow(data, cmap=None if data.ndim == 3 else "gray")
            ax.set_title(title, fontsize=14, color="#4fc3f7", fontweight="bold")
            ax.axis("off")
        fig.suptitle(f"Mask debug - {name}", fontsize=14, color="white")
        fig.tight_layout()
        out = OUT_DIR / f"debug_masks_{name.rsplit('.', 1)[0]}.png"
        fig.savefig(out, dpi=70)
        plt.close(fig)
        print("Sacuvano:", out.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:] or ["test1.jpg", "test4.jpg"])