"""
Camera calibration and distortion correction.

Sta radi:
  1. Pronalazi unutrasnje uglove sahovnice (9x6) na svim slikama iz camera_cal/
  2. Racuna matricu kamere (mtx) i distorzione koeficijente (dist)
  3. Cuva rezultat u camera_cal/calibration.json (da se ne racuna svaki put)
  4. Pravi slike za README u output_images/
"""
from pathlib import Path
import json
import re

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # crtanje bez otvaranja prozora
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CAL_DIR = ROOT / "camera_cal"
CAL_FILE = CAL_DIR / "calibration.json"
OUT_DIR = ROOT / "output_images"

NX, NY = 9, 6  # broj UNUTRASNJIH uglova po horizontali i vertikali


def _natural_key(path):
    """calibration2 ide pre calibration10."""
    m = re.search(r"\d+", path.stem)
    return int(m.group()) if m else 0


def find_corners(images_dir=CAL_DIR, nx=NX, ny=NY):
    """Vraca (objpoints, imgpoints, img_size, used, failed)."""
    # 3D tacke sahovnice u njenom koordinatnom sistemu: (0,0,0), (1,0,0) ... (8,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints, used, failed = [], [], [], []
    img_size = None

    for path in sorted(images_dir.glob("calibration*.jpg"), key=_natural_key):
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (sirina, visina)

        found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if found:
            # subpikselno doterivanje polozaja uglova -> preciznija kalibracija
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            used.append(path.name)
        else:
            failed.append(path.name)

    return objpoints, imgpoints, img_size, used, failed


def calibrate(images_dir=CAL_DIR, nx=NX, ny=NY, save=True, verbose=True):
    """Racuna mtx i dist, cuva u CAL_FILE."""
    objpoints, imgpoints, img_size, used, failed = find_corners(images_dir, nx, ny)
    if not objpoints:
        raise RuntimeError("Nijedna sahovnica nije pronadjena - proveri NX/NY.")

    rms, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    if verbose:
        print(f"Uglovi pronadjeni na {len(used)}/{len(used) + len(failed)} slika")
        if failed:
            print("Nisu pronadjeni na:", ", ".join(failed))
        print(f"Velicina slike: {img_size}")
        print(f"RMS reprojection error: {rms:.4f} px")
        print("Matrica kamere:\n", np.array2string(mtx, precision=2, suppress_small=True))
        print("Distorzioni koeficijenti (k1, k2, p1, p2, k3):\n",
              np.array2string(dist.ravel(), precision=4))

    if save:
        CAL_FILE.write_text(json.dumps(
            {"rms": rms, "mtx": mtx.tolist(), "dist": dist.ravel().tolist()}, indent=2))
        if verbose:
            print(f"Sacuvano u {CAL_FILE.relative_to(ROOT)}")
    return mtx, dist


def load_calibration(path=CAL_FILE):
    """Ucitava sacuvanu kalibraciju; ako ne postoji, kalibrise."""
    if Path(path).exists():
        data = json.loads(Path(path).read_text())
        return np.array(data["mtx"]), np.array(data["dist"])
    return calibrate()


def undistort(img, mtx, dist):
    """Ispravlja distorziju sociva."""
    return cv2.undistort(img, mtx, dist, None, mtx)


# ----------------------------------------------------------------------------
# Slike za README
# ----------------------------------------------------------------------------
def save_side_by_side(left, right, out_path, titles=("Before (distorted)", "After (undistorted)"),
                      source_name=""):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), facecolor="#1e1e1e")
    for ax, img, title, color in zip(axes, (left, right), titles, ("#ff8c42", "#4fc3f7")):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=18, color=color, fontweight="bold")
        ax.axis("off")
    if source_name:
        fig.suptitle(f"Distortion correction - {source_name}", fontsize=14, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def save_corners_example(img_name="calibration2.jpg", out_name="chessboard_corners.jpg"):
    img = cv2.imread(str(CAL_DIR / img_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
    if found:
        cv2.drawChessboardCorners(img, (NX, NY), corners, found)
        cv2.imwrite(str(OUT_DIR / out_name), img)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    mtx, dist = calibrate()

    save_corners_example()

    # Primer 1: sahovnica (distorzija je najocitija na ivicama)
    cal_img = cv2.imread(str(CAL_DIR / "calibration1.jpg"))
    save_side_by_side(cal_img, undistort(cal_img, mtx, dist),
                      OUT_DIR / "undistort_calibration.png", source_name="calibration1.jpg")

    # Primer 2: slika puta
    road = cv2.imread(str(ROOT / "test_images" / "test1.jpg"))
    save_side_by_side(road, undistort(road, mtx, dist),
                      OUT_DIR / "undistort_test1.png", source_name="test1.jpg")

    print("Slike sacuvane u output_images/")


if __name__ == "__main__":
    main()