"""
Thresholded binary image: 1 = verovatno piksel linije, 0 = ostalo.

Pokretanje (iz root foldera repoa):
    python -m src.thresholding

VERZIJA 2 (koristi je pipeline, funkcija threshold):
  - "svetla uska traka": morfoloski top-hat na L kanalu (HLS).
    Top-hat = slika - opening(slika). Opening horizontalnim kernelom sirine
    STRIPE_KERNEL px "obrise" sve svetle strukture uze od kernela, pa razlika
    ostavlja samo njih. Farbana linija je uska i svetlija od okoline -> prolazi.
    Ivica zida, granica asfalt/beton i tamne zakrpe NISU uske svetle trake
    -> ne prolaze (a Sobel ih je hvatao).
  - zuta: b kanal LAB prostora (plavo <-> zuto). Zuta farba ima b >= 150,
    a i svetao bez beton i siv asfalt ostaju ispod ~140.

VERZIJA 1 (threshold_v1, ostavljena radi poredjenja u README-u):
  Sobel x na L kanalu | S kanal | zuta i bela u HLS.
  Na challenge videima Sobel hvata ivicu betonske ograde i tamne zakrpe.

main() pravi:
  - binarne slike test slika (output_images/binary/)
  - slike za README (output_images/binary_<ime>.png)
  - poredjenje v1/v2 u birds-eye prikazu za frejmove iz output_images/frames/
    (output_images/compare/), ako su izvuceni sa src.extract_frames
"""
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
FRAMES_DIR = OUT_DIR / "frames"
COMPARE_DIR = OUT_DIR / "compare"

# ----------------------------------------------------------------------------
# Verzija 2 - parametri
# ----------------------------------------------------------------------------
STRIPE_KERNEL = 31         # px; linije uze od ovoga (u slici kamere) prolaze top-hat
STRIPE_THRESH = 20         # koliko linija mora biti svetlija od okoline (L, 0-255)
YELLOW_B_THRESH = 150      # LAB b kanal (128 = neutralno, vise = zuce)

# ----------------------------------------------------------------------------
# Verzija 1 - parametri
# ----------------------------------------------------------------------------
SOBEL_KERNEL = 3
SOBEL_X_THRESH = (30, 100)   # jacina vertikalne ivice (0-255 posle skaliranja)
S_THRESH = (170, 255)        # saturacija
S_MIN_LIGHTNESS = 100        # tamne senke imaju visok S -> S maska samo gde je L dovoljno visok
# HLS u OpenCV-u: H 0-179, L 0-255, S 0-255
YELLOW_LOW = (15, 80, 130)   # (H, L, S); S >= 130 da svetao bez beton ne prolazi kao zuta
YELLOW_HIGH = (35, 204, 255)
WHITE_LOW = (0, 200, 0)
WHITE_HIGH = (179, 255, 255)

CAMERA_SHAPE = (720, 1280)   # samo slike ove velicine su sa nase kamere
SHOWCASE = ["test5", "test4"]  # slike za README


# ----------------------------------------------------------------------------
# Verzija 2
# ----------------------------------------------------------------------------
def stripe_binary(img, ksize=STRIPE_KERNEL, thresh=STRIPE_THRESH):
    """Uske svetle trake (top-hat na L kanalu, horizontalni kernel)."""
    l_ch = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, 1))
    tophat = cv2.morphologyEx(l_ch, cv2.MORPH_TOPHAT, kernel)
    return (tophat >= thresh).astype(np.uint8)


def yellow_lab_binary(img, thresh=YELLOW_B_THRESH):
    """Zuta preko b kanala LAB prostora."""
    b_ch = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 2]
    return (b_ch >= thresh).astype(np.uint8)


def threshold(img, return_layers=False):
    """
    Glavna funkcija koju koristi pipeline (verzija 2).
    img: BGR slika (vec undistorted)
    Vraca binarnu sliku (0/1). Sa return_layers=True vraca i (traka, zuta).
    """
    stripe = stripe_binary(img)
    yellow = yellow_lab_binary(img)
    combined = stripe | yellow
    if return_layers:
        return combined, stripe, yellow
    return combined


# ----------------------------------------------------------------------------
# Verzija 1
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


def threshold_v1(img, return_layers=False):
    """Prva verzija: Sobel x | S kanal | zuta i bela (HLS)."""
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
def save_threshold_figure(img, stripe, yellow, combined, out_path, source_name):
    # zeleno = uska svetla traka, plavo = zuta
    stacked = np.dstack((np.zeros_like(stripe), stripe, yellow)) * 255

    fig, axes = plt.subplots(1, 3, figsize=(21, 5), facecolor="#1e1e1e")
    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Undistorted", "#ff8c42", None),
        (stacked, "Green = bright stripe, Blue = yellow", "#81c784", None),
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


def save_version_comparison(img, out_path, source_name):
    """Birds-eye: slika | binarna v1 | binarna v2."""
    from src.perspective import warp  # ovde, da se izbegne kruzni import

    panels = [
        (cv2.cvtColor(warp(img), cv2.COLOR_BGR2RGB), "Warped image", "#ff8c42", None),
        (warp(threshold_v1(img)), "Version 1 (Sobel + HLS)", "#e57373", "gray"),
        (warp(threshold(img)), "Version 2 (top-hat + LAB b)", "#4fc3f7", "gray"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 5), facecolor="#1e1e1e")
    for ax, (data, title, color_title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=16, color=color_title, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"Thresholding v1 vs v2 (birds-eye) - {source_name}", fontsize=14, color="white")
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

        combined, stripe, yellow = threshold(img, return_layers=True)
        cv2.imwrite(str(BINARY_DIR / path.name), combined * 255)

        if path.stem in SHOWCASE:
            save_threshold_figure(img, stripe, yellow, combined,
                                  OUT_DIR / f"binary_{path.stem}.png", path.name)

    print(f"Binarne slike sacuvane u {BINARY_DIR.relative_to(ROOT)}")
    print("Slike za README:", ", ".join(f"binary_{s}.png" for s in SHOWCASE))

    frames = sorted(FRAMES_DIR.glob("*.jpg")) if FRAMES_DIR.exists() else []
    if frames:
        COMPARE_DIR.mkdir(exist_ok=True)
        for path in frames:
            img = cv2.imread(str(path))
            if img.shape[:2] != CAMERA_SHAPE:
                continue
            save_version_comparison(undistort(img, mtx, dist),
                                    COMPARE_DIR / f"{path.stem}.png", path.name)
        print(f"Poredjenje v1/v2: {COMPARE_DIR.relative_to(ROOT)} ({len(frames)} frejmova)")


if __name__ == "__main__":
    main()