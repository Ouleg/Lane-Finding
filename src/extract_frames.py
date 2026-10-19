"""
Izvlaci pojedinacne (originalne, neobradjene) frejmove iz videa.

Pokretanje (iz root foldera repoa):
    python -m src.extract_frames challenge02 60 125 300

Snima output_images/frames/<video>_<frejm>.jpg
"""
import sys

import cv2

from src.calibration import ROOT

VIDEO_DIR = ROOT / "test_videos"
OUT_DIR = ROOT / "output_images" / "frames"


def main(video, frames):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{video}.mp4"))
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            print(f"Frejm {f} ne postoji")
            continue
        out = OUT_DIR / f"{video}_{f}.jpg"
        cv2.imwrite(str(out), img)
        print("Sacuvano:", out.relative_to(ROOT))
    cap.release()


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])