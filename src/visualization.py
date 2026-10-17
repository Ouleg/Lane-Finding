"""
Iscrtavanje detektovane trake nazad na originalnu sliku + tekst sa merenjima.

Koraci:
  1. U birds-eye prikazu nacrtamo povrsinu izmedju dva polinoma (zeleno)
     i same linije (narandzasto levo, plavo desno).
  2. unwarp (matrica Minv) vraca taj crtez u perspektivu kamere.
  3. cv2.addWeighted ga poluprovidno preklopi preko undistorted slike.
  4. Gore levo: tamni panel sa radijusom i pomerajem.
     Gore desno: mali birds-eye prikaz sa prozorima i polinomima (debug).

Ovde je i process_image(): ceo pipeline za jednu sliku
  undistort -> threshold -> warp -> find_lane -> metrics -> draw
"""
import cv2
import numpy as np

from src.calibration import ROOT, load_calibration, undistort
from src.tresholding import threshold, CAMERA_SHAPE
from src.perspective import warp, unwarp
from src.lane_detection import find_lane, eval_poly, sliding_window, draw_fit
from src.metrics import lane_metrics

TEST_DIR = ROOT / "test_images"
OUT_DIR = ROOT / "output_images"
FINAL_DIR = OUT_DIR / "final"

# boje (BGR)
LANE_FILL = (0, 200, 0)
LEFT_COLOR = (0, 140, 255)     # narandzasta
RIGHT_COLOR = (255, 160, 0)    # plava
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (40, 40, 40)

STRAIGHT_RADIUS_M = 3000       # iznad ovoga pisemo "straight"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_lane(undist, left_fit, right_fit, minv=None):
    """
    Zelena traka + linije, vraceno u perspektivu kamere i preklopljeno.
    minv: inverzna perspektivna matrica (default: ona iz perspective.py).
    """
    h, w = undist.shape[:2]
    ploty = np.arange(h)
    left_x = eval_poly(left_fit, ploty)
    right_x = eval_poly(right_fit, ploty)

    overlay = np.zeros_like(undist)
    left_pts = np.column_stack((left_x, ploty))
    right_pts = np.column_stack((right_x, ploty))[::-1]
    lane_poly = np.vstack((left_pts, right_pts)).astype(np.int32)
    cv2.fillPoly(overlay, [lane_poly], LANE_FILL)
    cv2.polylines(overlay, [left_pts.astype(np.int32)], False, LEFT_COLOR, 30)
    cv2.polylines(overlay, [right_pts.astype(np.int32)], False, RIGHT_COLOR, 30)

    overlay_cam = unwarp(overlay) if minv is None else warp(overlay, minv)
    return cv2.addWeighted(undist, 1.0, overlay_cam, 0.4, 0)


def format_metrics(m):
    if m["radius_m"] > STRAIGHT_RADIUS_M:
        radius_txt = f"Curve radius: straight (> {STRAIGHT_RADIUS_M} m)"
    else:
        radius_txt = f"Curve radius: {m['radius_m']:.0f} m"
    side = "right" if m["offset_m"] > 0 else "left"
    offset_txt = f"Offset: {abs(m['offset_m']):.2f} m {side} of center"
    return [radius_txt, offset_txt]


def draw_panel(img, lines, x=20, y=20, line_h=42, scale=1.0):
    """Tamni poluprovidni panel sa tekstom, gore levo."""
    out = img.copy()
    width = max(cv2.getTextSize(t, FONT, scale, 2)[0][0] for t in lines) + 30
    height = line_h * len(lines) + 20
    panel = out.copy()
    cv2.rectangle(panel, (x, y), (x + width, y + height), PANEL_COLOR, -1)
    out = cv2.addWeighted(panel, 0.65, out, 0.35, 0)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (x + 15, y + 40 + i * line_h), FONT, scale,
                    TEXT_COLOR, 2, cv2.LINE_AA)
    return out


def draw_inset(img, inset, scale=0.25, margin=20):
    """Mali birds-eye prikaz u gornjem desnom uglu."""
    out = img.copy()
    small = cv2.resize(inset, None, fx=scale, fy=scale)
    h, w = small.shape[:2]
    x0 = out.shape[1] - w - margin
    out[margin:margin + h, x0:x0 + w] = small
    cv2.rectangle(out, (x0, margin), (x0 + w, margin + h), TEXT_COLOR, 2)
    cv2.putText(out, "birds-eye", (x0 + 8, margin + h + 25), FONT, 0.6,
                TEXT_COLOR, 1, cv2.LINE_AA)
    return out


def process_image(img, mtx, dist, show_inset=True):
    """
    Ceo pipeline za jednu BGR sliku 1280x720.
    Vraca (izlazna slika, lane dict, metrics dict ili None).
    """
    undist = undistort(img, mtx, dist)
    binary_warped = warp(threshold(undist))
    lane = find_lane(binary_warped)

    if lane["left_fit"] is None or lane["right_fit"] is None:
        out = draw_panel(undist, ["Lane not detected"])
        return out, lane, None

    m = lane_metrics(lane["left_fit"], lane["right_fit"], undist.shape)
    out = draw_lane(undist, lane["left_fit"], lane["right_fit"])
    out = draw_panel(out, format_metrics(m))
    if show_inset:
        _, windows = sliding_window(binary_warped, return_windows=True)
        out = draw_inset(out, draw_fit(binary_warped, windows, lane))
    return out, lane, m


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    mtx, dist = load_calibration()

    for path in sorted(TEST_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img.shape[:2] != CAMERA_SHAPE:
            continue
        out, _, m = process_image(img, mtx, dist)
        cv2.imwrite(str(FINAL_DIR / path.name), out)
        status = " | ".join(format_metrics(m)) if m else "lane not detected"
        print(f"{path.name:22s} {status}")

        if path.stem == "test5":
            cv2.imwrite(str(OUT_DIR / "final_test5.jpg"), out)

    print(f"\nSlike: {FINAL_DIR.relative_to(ROOT)}  i  output_images/final_test5.jpg")


if __name__ == "__main__":
    main()