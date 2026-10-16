"""
Detekcija piksela linija i fitovanje polinoma 2. reda.

Ulaz: binarna slika u birds-eye prikazu (threshold -> warp).

Koraci:
  1. HISTOGRAM donje polovine slike: za svaku kolonu (x) saberemo bele piksele.
     Linije su skoro vertikalne, pa daju dva jasna vrha -> pocetne x pozicije
     leve i desne linije.
  2. SLIDING WINDOW: od dna ka vrhu slike pomeramo 9 prozora (sirine 2*MARGIN)
     za svaku liniju. Pikseli unutar prozora pripadaju liniji. Ako ih ima
     dovoljno (MINPIX), sledeci prozor centriramo na njihov prosecan x,
     pa prozori "prate" liniju i kad zakrivi.
  3. POLYFIT: kroz pronadjene piksele provucemo x = A*y^2 + B*y + C.
     Fitujemo x(y), a ne y(x), jer je linija skoro vertikalna (za isto x
     postoji vise y vrednosti).
  4. SEARCH AROUND POLY (za video): kad vec imamo polinom iz prethodnog
     frejma, ne trazimo ispocetka nego samo u pojasu +-MARGIN oko njega.
"""
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.calibration import ROOT, load_calibration, undistort
from src.tresholding import threshold, CAMERA_SHAPE
from src.perspective import warp

TEST_DIR = ROOT / "test_images"
OUT_DIR = ROOT / "output_images"
FIT_DIR = OUT_DIR / "lane_fit"

# ----------------------------------------------------------------------------
# Parametri
# ----------------------------------------------------------------------------
N_WINDOWS = 9        # broj prozora po visini slike
MARGIN = 100         # pola sirine prozora (px)
MINPIX = 50          # min broj piksela da bi se prozor pomerio
MIN_LINE_PIXELS = 300  # ispod ovoga smatramo da linija nije pronadjena
# Histogram trazi vrhove samo u ovom opsegu x. Ivica ograde (x < 100) i
# desna ivica slike se ignorisu; linije su posle warpa oko x=320 i x=960.
HIST_X_RANGE = (100, 1180)


# ----------------------------------------------------------------------------
# 1. Histogram
# ----------------------------------------------------------------------------
def lane_histogram(binary_warped):
    """Zbir belih piksela po kolonama, samo donja polovina slike."""
    h = binary_warped.shape[0]
    return np.sum(binary_warped[h // 2:, :], axis=0)


def find_base_positions(histogram):
    """x pozicije vrhova histograma levo i desno od sredine."""
    x_min, x_max = HIST_X_RANGE
    mid = histogram.shape[0] // 2
    left_base = x_min + int(np.argmax(histogram[x_min:mid]))
    right_base = mid + int(np.argmax(histogram[mid:x_max]))
    return left_base, right_base


# ----------------------------------------------------------------------------
# 2. Sliding window
# ----------------------------------------------------------------------------
def sliding_window(binary_warped, return_windows=False):
    """
    Vraca (leftx, lefty, rightx, righty) - koordinate piksela svake linije.
    Sa return_windows=True vraca i listu prozora za crtanje.
    """
    h = binary_warped.shape[0]
    window_h = h // N_WINDOWS
    nonzeroy, nonzerox = binary_warped.nonzero()

    leftx_current, rightx_current = find_base_positions(lane_histogram(binary_warped))
    left_idx, right_idx, windows = [], [], []

    for i in range(N_WINDOWS):
        y_low = h - (i + 1) * window_h
        y_high = h - i * window_h
        in_rows = (nonzeroy >= y_low) & (nonzeroy < y_high)

        for side, x_current in (("L", leftx_current), ("R", rightx_current)):
            x_low, x_high = x_current - MARGIN, x_current + MARGIN
            windows.append((x_low, y_low, x_high, y_high))
            good = np.flatnonzero(in_rows & (nonzerox >= x_low) & (nonzerox < x_high))
            if side == "L":
                left_idx.append(good)
                if len(good) > MINPIX:
                    leftx_current = int(np.mean(nonzerox[good]))
            else:
                right_idx.append(good)
                if len(good) > MINPIX:
                    rightx_current = int(np.mean(nonzerox[good]))

    left_idx = np.concatenate(left_idx)
    right_idx = np.concatenate(right_idx)
    pixels = (nonzerox[left_idx], nonzeroy[left_idx], nonzerox[right_idx], nonzeroy[right_idx])
    if return_windows:
        return pixels, windows
    return pixels


# ----------------------------------------------------------------------------
# 3. Polinom
# ----------------------------------------------------------------------------
def fit_line(x, y):
    """x = A*y^2 + B*y + C. Vraca [A, B, C] ili None ako nema dovoljno piksela."""
    if len(x) < MIN_LINE_PIXELS:
        return None
    return np.polyfit(y, x, 2)


def eval_poly(fit, ploty):
    return fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]


def find_lane(binary_warped, prev_left=None, prev_right=None):
    """
    Glavna funkcija za pipeline.
    Ako postoje polinomi iz prethodnog frejma -> search_around_poly,
    inace (ili ako ta pretraga ne uspe) -> sliding_window.
    Vraca dict sa fitovima i pikselima.
    """
    method = "sliding_window"
    if prev_left is not None and prev_right is not None:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped, prev_left, prev_right)
        method = "search_around_poly"
        if len(leftx) < MIN_LINE_PIXELS or len(rightx) < MIN_LINE_PIXELS:
            leftx, lefty, rightx, righty = sliding_window(binary_warped)
            method = "sliding_window"
    else:
        leftx, lefty, rightx, righty = sliding_window(binary_warped)

    return {
        "left_fit": fit_line(leftx, lefty),
        "right_fit": fit_line(rightx, righty),
        "leftx": leftx, "lefty": lefty,
        "rightx": rightx, "righty": righty,
        "method": method,
    }


# ----------------------------------------------------------------------------
# 4. Pretraga oko prethodnog polinoma (video)
# ----------------------------------------------------------------------------
def search_around_poly(binary_warped, left_fit, right_fit, margin=MARGIN):
    nonzeroy, nonzerox = binary_warped.nonzero()
    left_center = eval_poly(left_fit, nonzeroy)
    right_center = eval_poly(right_fit, nonzeroy)
    left_mask = np.abs(nonzerox - left_center) < margin
    right_mask = np.abs(nonzerox - right_center) < margin
    return (nonzerox[left_mask], nonzeroy[left_mask],
            nonzerox[right_mask], nonzeroy[right_mask])


# ----------------------------------------------------------------------------
# Slike za README
# ----------------------------------------------------------------------------
def draw_fit(binary_warped, windows=None, lane=None):
    """Prozori zeleno, levi pikseli crveno, desni plavo, polinomi zuto (BGR)."""
    out = np.dstack([binary_warped * 255] * 3).astype(np.uint8)
    if windows:
        for x1, y1, x2, y2 in windows:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    if lane is not None:
        out[lane["lefty"], lane["leftx"]] = (0, 0, 255)
        out[lane["righty"], lane["rightx"]] = (255, 0, 0)
        ploty = np.arange(binary_warped.shape[0])
        for fit in (lane["left_fit"], lane["right_fit"]):
            if fit is not None:
                pts = np.column_stack((eval_poly(fit, ploty), ploty)).astype(np.int32)
                cv2.polylines(out, [pts], False, (0, 255, 255), 6)
    return out


def save_histogram_figure(binary_warped, out_img, out_path, source_name):
    hist = lane_histogram(binary_warped)
    left_base, right_base = find_base_positions(hist)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), facecolor="#1e1e1e")
    ax = axes[0]
    ax.set_facecolor("#1e1e1e")
    ax.plot(hist, color="#4fc3f7", linewidth=2)
    for x in (left_base, right_base):
        ax.axvline(x, color="#ff8c42", linestyle="--")
    ax.axvspan(0, HIST_X_RANGE[0], color="gray", alpha=0.3)
    ax.axvspan(HIST_X_RANGE[1], hist.shape[0], color="gray", alpha=0.3)
    ax.set_title(f"Histogram (bottom half)  peaks: x={left_base}, x={right_base}",
                 fontsize=15, color="#ff8c42", fontweight="bold")
    ax.set_xlim(0, hist.shape[0])
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    axes[1].imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Sliding windows + polynomial fit", fontsize=15,
                      color="#4fc3f7", fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(f"Lane detection - {source_name}", fontsize=14, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    mtx, dist = load_calibration()

    print(f"{'slika':22s} {'levo px':>8s} {'desno px':>9s}   levi fit [A, B, C]  |  desni fit [A, B, C]")
    for path in sorted(TEST_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img.shape[:2] != CAMERA_SHAPE:
            continue
        binary_warped = warp(threshold(undistort(img, mtx, dist)))

        (leftx, lefty, rightx, righty), windows = sliding_window(binary_warped, return_windows=True)
        lane = {
            "left_fit": fit_line(leftx, lefty), "right_fit": fit_line(rightx, righty),
            "leftx": leftx, "lefty": lefty, "rightx": rightx, "righty": righty,
        }
        out_img = draw_fit(binary_warped, windows, lane)
        cv2.imwrite(str(FIT_DIR / path.name), out_img)

        fmt = lambda f: "NIJE NADJENA" if f is None else np.array2string(f, precision=5, suppress_small=False)
        print(f"{path.name:22s} {len(leftx):8d} {len(rightx):9d}   {fmt(lane['left_fit'])}  |  {fmt(lane['right_fit'])}")

        if path.stem == "test5":
            save_histogram_figure(binary_warped, out_img, OUT_DIR / "lane_fit_test5.png", path.name)

    print(f"\nSlike: {FIT_DIR.relative_to(ROOT)}  i  output_images/lane_fit_test5.png")


if __name__ == "__main__":
    main()