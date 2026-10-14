"""
Thresholded binary image: color + gradient.

Sta radi:
  1. Ucita kalibraciju i ispravi distorziju test slike
  2. Prebaci sliku u HLS prostor boja (Hue, Lightness, Saturation)
  3. Napravi tri maske:
       - gradijent: Sobel po x osi na L kanalu (vertikalne ivice = linije)
       - saturacija: S kanal (jake boje, linije ostaju vidljive i u senci)
       - boja: zuta i bela (direktno boje linija na putu)
  4. Spoji maske (OR) u jednu binarnu sliku: 1 = verovatno linija, 0 = ostalo
  5. Sacuva binarne slike i slike za README u output_images/
"""
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.calibration import ROOT, load_calibration, undistort

TEST_DIR = ROOT / "test_images"
OUT_DIR = ROOT / "output_images"
BINARY_DIR = OUT_DIR / "binary"

# ----------------------------------------------------------------------------
# Pragovi - ovde se podesava ponasanje
# ----------------------------------------------------------------------------
SOBEL_KERNEL = 3
SOBEL_X_THRESH = (30, 100)   # jacina vertikalne ivice (0-255 posle skaliranja)
S_THRESH = (170, 255)        # saturacija
S_MIN_LIGHTNESS = 100        # tamne senke imaju visok S -> S maska samo gde je L dovoljno visok
# HLS u OpenCV-u: H 0-179, L 0-255, S 0-255
YELLOW_LOW = (15, 80, 130)    # (H, L, S) S >= 130 je moralo da svetao bez beton ne prolazi kao zuta
YELLOW_HIGH = (35, 204, 255)
WHITE_LOW = (0, 200, 0)
WHITE_HIGH = (179, 255, 255)

CAMERA_SHAPE = (720, 1280)   # samo slike ove velicine su sa nase kamere
SHOWCASE = ["test5", "challange00101"]  # slike za README


# ----------------------------------------------------------------------------
# Maske
# ----------------------------------------------------------------------------
def sobel_x_binary(channel, ksize=SOBEL_KERNEL, thresh=SOBEL_X_THRESH):
    """Gradijent po x: hvata vertikalne ivice, kakve su linije trake."""
    sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize))
    max_val = sobel.max()
    if max_val == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    scaled = np.uint8(255 * sobel / max_val)
    return ((scaled >= thresh[0]) & (scaled <= thresh[1])).astype(np.uint8)


def channel_binary(channel, thresh):
    """1 gde je vrednost kanala u opsegu thresh."""
    return ((channel >= thresh[0]) & (channel <= thresh[1])).astype(np.uint8)


def yellow_white_binary(hls):
    """Direktno trazi zute i bele piksele."""
    yellow = cv2.inRange(hls, YELLOW_LOW, YELLOW_HIGH)
    white = cv2.inRange(hls, WHITE_LOW, WHITE_HIGH)
    return ((yellow | white) > 0).astype(np.uint8)


def threshold(img, return_layers=False):
    """
    Glavna funkcija koju koristi pipeline.
    img: BGR slika (vec undistorted)
    Vraca binarnu sliku (0/1). Sa return_layers=True vraca i (gradijent, boja).
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_ch, s_ch = hls[:, :, 1], hls[:, :, 2]

    grad = sobel_x_binary(l_ch)
    saturated = channel_binary(s_ch, S_THRESH) & (l_ch >= S_MIN_LIGHTNESS)
    color = saturated | yellow_white_binary(hls)
    combined = grad | color

    if return_layers:
        return combined, grad, color
    return combined


# ----------------------------------------------------------------------------
# Slike za README
# ----------------------------------------------------------------------------
def save_threshold_figure(img, grad, color, combined, out_path, source_name):
    # zeleno = doprinos gradijenta, plavo = doprinos boje
    stacked = np.dstack((np.zeros_like(grad), grad, color)) * 255

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), facecolor="#1e1e1e")
    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Undistorted", "#ff8c42", None),
        (stacked, "Green = gradient, Blue = color", "#81c784", None),
        (combined, "Combined binary", "#4fc3f7", "gray"),
    ]
    for ax, (data, title, color_title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=16, color=color_title, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"Thresholding - {source_name}", fontsize=14, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    BINARY_DIR.mkdir(parents=True, exist_ok=True)
    mtx, dist = load_calibration()

    for path in sorted(TEST_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img.shape[:2] == CAMERA_SHAPE:
            img = undistort(img, mtx, dist)
        # slike druge velicine su sa druge kamere -> nasa kalibracija ne vazi

        combined, grad, color = threshold(img, return_layers=True)
        cv2.imwrite(str(BINARY_DIR / path.name), combined * 255)

        if path.stem in SHOWCASE:
            save_threshold_figure(img, grad, color, combined,
                                  OUT_DIR / f"binary_{path.stem}.png", path.name)

    print(f"Binarne slike sacuvane u {BINARY_DIR.relative_to(ROOT)}")
    print("Slike za README:", ", ".join(f"binary_{s}.png" for s in SHOWCASE))


if __name__ == "__main__":
    main()