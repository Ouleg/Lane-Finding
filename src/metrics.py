"""
Radijus zakrivljenosti trake i pozicija vozila u odnosu na centar trake.

Ideja:
  Polinomi iz lane_detection su u PIKSELIMA birds-eye slike.
  Da bismo dobili metre, koristimo poznate dimenzije puta (US autoput):
    - sirina trake = 3.7 m  -> u birds-eye slici ~640 px (DST x: 320 -> 960)
    - isprekidana linija: crta 3 m + razmak 9 m = 12 m po periodi
      -> vidljiv deo puta u birds-eye slici je ~30 m na 720 px
  Koeficijenti polinoma se preracunaju u metre, pa radijus po formuli
      R = (1 + (2*A*y + B)^2)^(3/2) / |2*A|
  racunamo na dnu slike (y = visina), tj. tacno ispred vozila.

  Pomeraj: kamera je na sredini vozila, pa je centar vozila = sredina slike.
  Centar trake = sredina izmedju leve i desne linije na dnu slike.
      offset = (centar vozila - centar trake) * XM_PER_PIX
  offset > 0  -> vozilo je DESNO od centra
  offset < 0  -> vozilo je LEVO od centra
"""
import cv2
import numpy as np

from src.calibration import ROOT, load_calibration, undistort
from src.tresholding import threshold, CAMERA_SHAPE
from src.perspective import warp
from src.lane_detection import find_lane, eval_poly

TEST_DIR = ROOT / "test_images"

LANE_WIDTH_M = 3.7
LANE_WIDTH_PX = 640            # DST: 960 - 320
XM_PER_PIX = LANE_WIDTH_M / LANE_WIDTH_PX
YM_PER_PIX = 30 / 720          # ~30 m vidljivog puta na 720 px


def poly_to_meters(fit):
    """[A, B, C] u pikselima -> [A, B, C] u metrima (x i y u metrima)."""
    a, b, c = fit
    return np.array([
        a * XM_PER_PIX / YM_PER_PIX ** 2,
        b * XM_PER_PIX / YM_PER_PIX,
        c * XM_PER_PIX,
    ])


def radius_of_curvature(fit, y_eval_px):
    """Radijus u metrima u tacki y_eval_px (u pikselima)."""
    a, b, _ = poly_to_meters(fit)
    y = y_eval_px * YM_PER_PIX
    return (1 + (2 * a * y + b) ** 2) ** 1.5 / max(abs(2 * a), 1e-9)


def lane_metrics(left_fit, right_fit, img_shape):
    """
    Glavna funkcija za pipeline. img_shape = (visina, sirina).
    Vraca dict sa vrednostima u metrima.
    """
    h, w = img_shape[:2]
    y_bottom = h - 1
    left_x = eval_poly(left_fit, y_bottom)
    right_x = eval_poly(right_fit, y_bottom)

    lane_center = (left_x + right_x) / 2
    car_center = w / 2

    left_r = radius_of_curvature(left_fit, y_bottom)
    right_r = radius_of_curvature(right_fit, y_bottom)

    return {
        "left_radius_m": left_r,
        "right_radius_m": right_r,
        "radius_m": (left_r + right_r) / 2,
        "offset_m": (car_center - lane_center) * XM_PER_PIX,
        "lane_width_m": (right_x - left_x) * XM_PER_PIX,
        # rastojanje od centra vozila do svake linije (za upozorenje, korak 8)
        "dist_left_m": (car_center - left_x) * XM_PER_PIX,
        "dist_right_m": (right_x - car_center) * XM_PER_PIX,
    }


def describe_offset(offset_m):
    side = "right" if offset_m > 0 else "left"
    return f"Vehicle is {abs(offset_m):.2f}m {side} of center"


def main():
    mtx, dist = load_calibration()
    print(f"{'slika':22s} {'R levo':>9s} {'R desno':>9s} {'R sred.':>9s} "
          f"{'offset':>8s} {'sirina':>7s}")
    print(f"{'':22s} {'[m]':>9s} {'[m]':>9s} {'[m]':>9s} {'[m]':>8s} {'[m]':>7s}")

    for path in sorted(TEST_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img.shape[:2] != CAMERA_SHAPE:
            continue
        binary_warped = warp(threshold(undistort(img, mtx, dist)))
        lane = find_lane(binary_warped)
        if lane["left_fit"] is None or lane["right_fit"] is None:
            print(f"{path.name:22s}  linija nije pronadjena")
            continue

        m = lane_metrics(lane["left_fit"], lane["right_fit"], binary_warped.shape)
        print(f"{path.name:22s} {m['left_radius_m']:9.0f} {m['right_radius_m']:9.0f} "
              f"{m['radius_m']:9.0f} {m['offset_m']:+8.2f} {m['lane_width_m']:7.2f}")

    print("\nOcekivano: straight_lines -> veliki radijus (> 2000 m), sirina ~3.7 m,")
    print("krivine na test slikama -> radijus reda velicine 500-1500 m.")


if __name__ == "__main__":
    main()