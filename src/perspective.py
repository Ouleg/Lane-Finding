"""
Perspective transform: pogled kamere -> pogled odozgo ("birds-eye view").

Ideja:
  Na slici kamere paralelne linije trake se skupljaju ka horizontu.
  Izaberemo 4 tacke na putu koje u stvarnosti cine pravougaonik (SRC trapez)
  i kazemo OpenCV-u gde zelimo da budu na novoj slici (DST pravougaonik).
  cv2.getPerspectiveTransform iz ta 4 para tacaka racuna matricu 3x3 (M),
  a cv2.warpPerspective tom matricom "rasteze" celu sliku.
  Inverzna matrica (Minv) vraca rezultat nazad na sliku kamere (koristimo na kraju).

"""
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.calibration import ROOT, load_calibration, undistort
from src.tresholding import threshold, CAMERA_SHAPE

TEST_DIR = ROOT / "test_images"
OUT_DIR = ROOT / "output_images"
WARPED_DIR = OUT_DIR / "warped_binary"

# ----------------------------------------------------------------------------
# Tacke za 1280x720 sliku. Redosled: gore-levo, gore-desno, dole-desno, dole-levo
# SRC su izabrane na straight_lines1.jpg tako da leze na linijama trake.
# ----------------------------------------------------------------------------
SRC = np.float32([
    [585, 460],
    [695, 460],
    [1127, 720],
    [203, 720],
])
DST = np.float32([
    [320, 0],
    [960, 0],
    [960, 720],
    [320, 720],
])

M = cv2.getPerspectiveTransform(SRC, DST)
MINV = cv2.getPerspectiveTransform(DST, SRC)


def warp(img, matrix=M):
    """Pogled kamere -> birds-eye."""
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR)


def unwarp(img):
    """Birds-eye -> pogled kamere."""
    return warp(img, MINV)


# ----------------------------------------------------------------------------
# Slike za README
# ----------------------------------------------------------------------------
def _draw_polygon(img, pts, color=(0, 0, 255), thickness=3):
    out = img.copy()
    cv2.polylines(out, [pts.astype(np.int32)], isClosed=True, color=color, thickness=thickness)
    return out


def _save_pair(left, right, titles, out_path, source_name, cmap=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), facecolor="#1e1e1e")
    for ax, img, title, color in zip(axes, (left, right), titles, ("#ff8c42", "#4fc3f7")):
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=18, color=color, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"Perspective transform - {source_name}", fontsize=14, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    WARPED_DIR.mkdir(parents=True, exist_ok=True)
    mtx, dist = load_calibration()

    # 1) Provera na pravom putu: crveni trapez -> crveni pravougaonik
    for name in ("straight_lines1", "straight_lines2"):
        img = undistort(cv2.imread(str(TEST_DIR / f"{name}.jpg")), mtx, dist)
        before = _draw_polygon(img, SRC)
        after = _draw_polygon(warp(img), DST)
        _save_pair(before, after, ("Undistorted + source points", "Warped + destination points"),
                   OUT_DIR / f"warped_{name}.png", f"{name}.jpg")

    # 2) Binarne slike u birds-eye prikazu
    for path in sorted(TEST_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img.shape[:2] != CAMERA_SHAPE:
            # druga kamera -> tacke i kalibracija ne vaze
            print(f"Preskocena {path.name}: velicina {img.shape[1]}x{img.shape[0]}")
            continue
        img = undistort(img, mtx, dist)
        binary = threshold(img)
        warped = warp(binary)
        cv2.imwrite(str(WARPED_DIR / path.name), warped * 255)

        if path.stem == "test5":
            _save_pair(binary, warped, ("Binary", "Warped binary (birds-eye)"),
                       OUT_DIR / "warped_binary_test5.png", path.name, cmap="gray")

    print("Provera:  output_images/warped_straight_lines1.png, warped_straight_lines2.png")
    print("Primer:   output_images/warped_binary_test5.png")
    print(f"Sve slike: {WARPED_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()