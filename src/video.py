"""
Obrada videa: pipeline frejm po frejm + pracenje linija kroz vreme.

Pokretanje (iz root foldera repoa):
    python -m src.video test_videos/project_video03.mp4
    python -m src.video all                      # svih 6 videa
    python -m src.video test_videos/project_video01.mp4 --check   # provera perspektive
    python -m src.video test_videos/challenge01.mp4 --max-frames 200

Izlaz:
    output_videos/<ime>.mp4        obradjen video
    output_videos/<ime>_log.csv    vrednosti za svaki frejm (radijus, offset, ...)

Sta je novo u odnosu na pojedinacne slike:
  1. PRACENJE: kad u frejmu nadjemo linije, u sledecem ih trazimo samo oko
     prethodnih polinoma (search_around_poly) umesto sliding window ispocetka.
  2. PROVERA (sanity check): rezultat frejma prihvatamo samo ako
       - su obe linije pronadjene
       - je sirina trake na dnu izmedju 2.8 i 4.5 m
       - su linije priblizno paralelne (sirina gore i dole se razlikuje < 1.0 m)
  3. USREDNJAVANJE: crtamo prosek poslednjih N prihvacenih polinoma.
     Tako traka ne "skace" iz frejma u frejm.
  4. Ako provera ne prodje, a jedna linija ima dosta piksela i nalazi se
     tamo gde je bila u prethodnom frejmu, slabiju (obicno isprekidanu)
     zamenimo jacom pomerenom za sirinu trake.
  5. Ako provera i dalje ne prodje, crtamo prethodni prosek. Posle
     RESEARCH_AFTER losih frejmova trazimo ispocetka (sliding window),
     a tek posle FORGET_AFTER (~1 s) brisemo istoriju ("Lane not detected").

Dve kamere:
  1280x720 - kalibrisana kamera (test slike, challenge01-03, project_video03)
  960x540  - druga kamera (project_video01-02), nema sahovnica za nju:
             frejm se uveca na 1280x720, undistort se preskace i koriste se
             posebne SRC tacke za perspektivu.
  VIDEO_SRC    - posebne SRC tacke za pojedinacan video (npr. challenge02),
                 kad je kamera drugacije postavljena ili put ima drugi nagib.
"""
import argparse
import csv
from collections import deque

import cv2
import numpy as np

from src.calibration import ROOT, load_calibration, undistort
from src.tresholding import threshold
from src.perspective import SRC, DST, M, MINV, warp, _draw_polygon, _save_pair
from src.lane_detection import find_lane, eval_poly, draw_fit
from src.metrics import lane_metrics, XM_PER_PIX
from src.visualization import draw_lane, draw_panel, draw_inset, format_metrics

VIDEO_DIR = ROOT / "test_videos"
OUT_DIR = ROOT / "output_videos"

FRAME_SIZE = (1280, 720)          # (sirina, visina) u kojoj radi pipeline

# Kamera B (960x540), tacke u koordinatama UVECANOG frejma 1280x720.
# Izabrane na prvom frejmu project_video01 (prav put) - proveri sa --check.
SRC_B = np.float32([
    [553, 467],
    [737, 467],
    [1153, 720],
    [208, 720],
])


def make_camera(name, do_undistort, src):
    src = np.float32(src)
    return {"name": name, "undistort": do_undistort, "src": src,
            "M": cv2.getPerspectiveTransform(src, DST),
            "MINV": cv2.getPerspectiveTransform(DST, src)}


CAMERAS = {
    (720, 1280): make_camera("A (calibrated)", True, SRC),
    (540, 960): make_camera("B (uncalibrated)", False, SRC_B),
}

# Isti senzor kao kamera A, ali drugacije postavljena (nagib/visina) ili put
# ima drugi nagib: sa SRC tackama kamere A se linije u birds-eye prikazu
# skupljaju ka vrhu (~1 m razlike u sirini). Tacke procenjene sa
# --fit-src na frejmu 475 (skoro prav put), pa provereno sa --check.
VIDEO_SRC = {
    "challenge02": [[610, 476], [721, 476], [1124, 720], [293, 720]],
}


def get_camera(frame_shape, video_name):
    base = CAMERAS.get(frame_shape[:2])
    if base is None:
        raise ValueError(f"Nepoznata rezolucija {frame_shape[1]}x{frame_shape[0]}")
    if video_name in VIDEO_SRC:
        return make_camera(f"{base['name']}, custom SRC", base["undistort"], VIDEO_SRC[video_name])
    return base

# Pracenje
HISTORY = 8               # koliko poslednjih dobrih frejmova usrednjavamo
RESEARCH_AFTER = 5        # posle ovoliko losih frejmova -> sliding window ispocetka
FORGET_AFTER = 25         # posle ovoliko losih frejmova (~1 s) -> brisemo istoriju
MIN_WIDTH_M, MAX_WIDTH_M = 2.8, 4.5
MAX_WIDTH_DIFF_M = 1.0    # razlika sirine trake gore/dole (paralelnost)
MIN_STRONG_PIXELS = 3000  # linija sa bar ovoliko piksela moze da "spase" slabiju
MAX_JUMP_M = 0.5          # jaca linija ne sme da skoci vise od ovoga od prethodnog frejma


def lane_widths(left_fit, right_fit, h):
    """Sirina trake u metrima na dnu i na vrhu birds-eye slike."""
    w_bottom = (eval_poly(right_fit, h - 1) - eval_poly(left_fit, h - 1)) * XM_PER_PIX
    w_top = (eval_poly(right_fit, 0) - eval_poly(left_fit, 0)) * XM_PER_PIX
    return w_bottom, w_top


class LaneTracker:
    def __init__(self):
        self.left_hist = deque(maxlen=HISTORY)
        self.right_hist = deque(maxlen=HISTORY)
        self.misses = 0

    @property
    def left_fit(self):
        return np.mean(self.left_hist, axis=0) if self.left_hist else None

    @property
    def right_fit(self):
        return np.mean(self.right_hist, axis=0) if self.right_hist else None

    @staticmethod
    def sanity_check(left_fit, right_fit, h):
        if left_fit is None or right_fit is None:
            return False, "line missing"
        w_bottom, w_top = lane_widths(left_fit, right_fit, h)
        if not MIN_WIDTH_M <= w_bottom <= MAX_WIDTH_M:
            return False, f"width {w_bottom:.2f} m"
        if abs(w_top - w_bottom) > MAX_WIDTH_DIFF_M:
            return False, "not parallel"
        return True, "ok"

    def rescue_weak_line(self, lane, h):
        """
        Provera nije prosla -> verovatno je jedna linija (obicno isprekidana
        desna) lose fitovana ili nije nadjena. Slabiju liniju (manje piksela)
        zamenimo jacom, pomerenom za sirinu trake iz istorije.
        Jaca linija mora biti na dnu slike blizu svoje prethodne pozicije
        (< MAX_JUMP_M), da ne bismo "spasavali" pogresnom linijom.
        Vraca (left, right, razlog) ili None.
        """
        if not self.left_hist:
            return None
        n_left, n_right = len(lane["leftx"]), len(lane["rightx"])
        prev_l, prev_r = self.left_fit, self.right_fit
        width_px = eval_poly(prev_r, h - 1) - eval_poly(prev_l, h - 1)
        max_jump_px = MAX_JUMP_M / XM_PER_PIX

        def close(fit, prev):
            return abs(eval_poly(fit, h - 1) - eval_poly(prev, h - 1)) < max_jump_px

        left_strong = (lane["left_fit"] is not None and n_left >= MIN_STRONG_PIXELS
                       and close(lane["left_fit"], prev_l))
        right_strong = (lane["right_fit"] is not None and n_right >= MIN_STRONG_PIXELS
                        and close(lane["right_fit"], prev_r))

        if left_strong and (n_left >= n_right or not right_strong):
            left = lane["left_fit"].copy()
            right = left.copy()
            right[2] += width_px
            reason = "right line from left"
        elif right_strong:
            right = lane["right_fit"].copy()
            left = right.copy()
            left[2] -= width_px
            reason = "left line from right"
        else:
            return None
        ok, _ = self.sanity_check(left, right, h)
        return (left, right, reason) if ok else None

    def update(self, binary_warped):
        """Vraca (lane dict sa usrednjenim fitovima, prihvacen?, razlog)."""
        h = binary_warped.shape[0]
        research = self.misses >= RESEARCH_AFTER
        prev_l, prev_r = (None, None) if research else (self.left_fit, self.right_fit)
        lane = find_lane(binary_warped, prev_l, prev_r)

        ok, reason = self.sanity_check(lane["left_fit"], lane["right_fit"], h)
        if lane["left_fit"] is not None and lane["right_fit"] is not None:
            w_bottom, w_top = lane_widths(lane["left_fit"], lane["right_fit"], h)
            lane["raw_width_diff_m"] = w_top - w_bottom
        if not ok:
            rescued = self.rescue_weak_line(lane, h)
            if rescued:
                lane["left_fit"], lane["right_fit"], reason = rescued
                ok = True

        if ok:
            self.left_hist.append(lane["left_fit"])
            self.right_hist.append(lane["right_fit"])
            self.misses = 0
        else:
            self.misses += 1
            if self.misses >= FORGET_AFTER:
                self.left_hist.clear()
                self.right_hist.clear()
                reason += " -> lost"
            elif self.misses >= RESEARCH_AFTER:
                reason += " -> re-search"

        lane["left_fit"], lane["right_fit"] = self.left_fit, self.right_fit
        return lane, ok, reason


def prepare_frame(frame, mtx, dist, cam):
    """Uveca/undistortuje frejm prema kameri. Vraca frejm 1280x720."""
    if frame.shape[:2] != (FRAME_SIZE[1], FRAME_SIZE[0]):
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    if cam["undistort"]:
        frame = undistort(frame, mtx, dist)
    return frame


def process_frame(frame, tracker, mtx, dist, cam):
    img = prepare_frame(frame, mtx, dist, cam)
    binary_warped = warp(threshold(img), cam["M"])
    lane, ok, reason = tracker.update(binary_warped)

    if lane["left_fit"] is None or lane["right_fit"] is None:
        out = draw_panel(img, ["Lane not detected", f"status: {reason}"])
        return out, None, lane, ok, reason

    m = lane_metrics(lane["left_fit"], lane["right_fit"], img.shape)
    out = draw_lane(img, lane["left_fit"], lane["right_fit"], cam["MINV"])
    status = f"{lane['method'].replace('_', ' ')} | {reason}"
    out = draw_panel(out, format_metrics(m) + [f"status: {status}"])
    out = draw_inset(out, draw_fit(binary_warped, None, lane))
    return out, m, lane, ok, reason


def process_video(path, mtx, dist, max_frames=None):
    OUT_DIR.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = OUT_DIR / f"{path.stem}.mp4"
    log_path = OUT_DIR / f"{path.stem}_log.csv"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, FRAME_SIZE)

    tracker = LaneTracker()
    cam = None
    n = accepted = 0
    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "time_s", "accepted", "reason", "method", "left_px", "right_px",
                      "raw_width_diff_m", "radius_m", "offset_m", "lane_width_m",
                      "dist_left_m", "dist_right_m"])
        while True:
            ok_read, frame = cap.read()
            if not ok_read or (max_frames and n >= max_frames):
                break
            if cam is None:
                cam = get_camera(frame.shape, path.stem)
                print(f"{path.name}: kamera {cam['name']}")
            out, m, lane, ok, reason = process_frame(frame, tracker, mtx, dist, cam)
            writer.write(out)
            accepted += ok
            diff = lane.get("raw_width_diff_m")
            row = [n, f"{n / fps:.2f}", int(ok), reason, lane["method"],
                   len(lane["leftx"]), len(lane["rightx"]),
                   "" if diff is None else f"{diff:.2f}"]
            if m:
                row += [  # vrednosti iz usrednjenih (nacrtanih) fitova
f"{m['radius_m']:.0f}", f"{m['offset_m']:.3f}", f"{m['lane_width_m']:.2f}",
                        f"{m['dist_left_m']:.3f}", f"{m['dist_right_m']:.3f}"]
            log.writerow(row)
            n += 1
            if n % 100 == 0:
                print(f"  {n}/{total} frejmova")

    cap.release()
    writer.release()
    print(f"{path.name}: {n} frejmova, prihvaceno {accepted} ({100 * accepted / max(n, 1):.0f}%)")
    print(f"  -> {out_path.relative_to(ROOT)}, {log_path.relative_to(ROOT)}")


def read_frame(path, frame_idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    _, frame = cap.read()
    cap.release()
    return frame


def check_perspective(path, frame_idx=0):
    """Snima frejm sa SRC tackama i njegov birds-eye prikaz sa DST tackama."""
    OUT_DIR.mkdir(exist_ok=True)
    mtx, dist = load_calibration()
    frame = read_frame(path, frame_idx)
    cam = get_camera(frame.shape, path.stem)
    img = prepare_frame(frame, mtx, dist, cam)
    before = _draw_polygon(img, cam["src"])
    after = _draw_polygon(warp(img, cam["M"]), DST)
    out = OUT_DIR / f"check_{path.stem}.png"
    _save_pair(before, after, ("Frame + source points", "Warped + destination points"),
               out, f"{path.name}, frame {frame_idx}, camera {cam['name']}")
    print("Sacuvano:", out.relative_to(ROOT))


def fit_src(path, frame_idx, top_margin=38):
    """
    Procenjuje SRC tacke na frejmu gde je put PRAV:
      1. binarna slika (threshold) u trapezu ispred auta
      2. Hough transformacija nalazi duzi; leve (negativan nagib) i desne
         (pozitivan nagib) se fituju pravom x = a*y + b
      3. presek dve prave = tacka nedogleda (horizont)
      4. SRC: dno slike (y=720) i y = horizont + top_margin na obe prave
    """
    mtx, dist = load_calibration()
    frame = read_frame(path, frame_idx)
    cam = get_camera(frame.shape, path.stem)
    img = prepare_frame(frame, mtx, dist, cam)
    binary = (threshold(img) * 255).astype(np.uint8)
    roi = np.zeros_like(binary)
    cv2.fillPoly(roi, [np.array([[150, 690], [560, 470], [760, 470], [1200, 690]], np.int32)], 255)
    segs = cv2.HoughLinesP(cv2.bitwise_and(binary, roi), 1, np.pi / 180, 40,
                           minLineLength=40, maxLineGap=60)
    if segs is None:
        print("Hough nije nasao linije - probaj drugi frejm")
        return
    pts = {"L": ([], []), "R": ([], [])}
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(float):
        if y1 == y2:
            continue
        slope = (x2 - x1) / (y2 - y1)
        if 0.5 < abs(slope) < 3:
            xs, ys = pts["L" if slope < 0 else "R"]
            xs += [x1, x2]
            ys += [y1, y2]
    if not pts["L"][0] or not pts["R"][0]:
        print("Nije nadjena jedna od linija - probaj drugi frejm")
        return
    la, lb = np.polyfit(pts["L"][1], pts["L"][0], 1)
    ra, rb = np.polyfit(pts["R"][1], pts["R"][0], 1)
    y_vp = (rb - lb) / (la - ra)
    y_top = int(round(y_vp + top_margin))
    src = [[la * y_top + lb, y_top], [ra * y_top + rb, y_top], [ra * 720 + rb, 720], [la * 720 + lb, 720]]
    src = [[int(round(x)), int(y)] for x, y in src]
    print(f"{path.name}, frejm {frame_idx}: horizont y = {y_vp:.0f}")
    print(f"Predlog:  VIDEO_SRC['{path.stem}'] = {src}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="putanja do videa ili 'all'")
    parser.add_argument("--check", action="store_true", help="samo provera perspektive")
    parser.add_argument("--fit-src", action="store_true",
                        help="proceni SRC tacke na frejmu --frame (put mora biti prav)")
    parser.add_argument("--frame", type=int, default=0, help="frejm za --check / --fit-src")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(VIDEO_DIR.glob("*.mp4")) if args.video == "all" else [ROOT / args.video]
    if args.check or args.fit_src:
        for p in paths:
            if args.fit_src:
                fit_src(p, args.frame)
            if args.check:
                check_perspective(p, args.frame)
        return

    mtx, dist = load_calibration()
    for p in paths:
        process_video(p, mtx, dist, args.max_frames)


if __name__ == "__main__":
    main()