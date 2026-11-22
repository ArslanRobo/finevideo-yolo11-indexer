"""How many sampled frames are near-copies of the previous sampled frame?

    python tools/measure_duplicates.py <folder of .parquet> [--videos 40]

If many are, the model could be skipped for them and the previous frame's
detections reused. Compares each sampled frame (3 per second) with the
previous one from the same video on a small grayscale thumbnail, and
reports the share of frames below several difference thresholds.
CPU only.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from videoindex.decode import FrameSampler, frame_to_bgr, iter_frames  # noqa: E402

THRESHOLDS = [0.5, 1, 2, 3, 5]  # mean absolute difference, in grey levels (0-255)


def thumbnail(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--videos", type=int, default=40)
    args = parser.parse_args()

    diffs = []
    n_videos = 0
    for path in sorted(Path(args.input).glob("*.parquet")):
        column = pq.read_table(path, columns=["mp4"]).column("mp4")
        for row in range(len(column)):
            if n_videos >= args.videos:
                break
            n_videos += 1
            sampler, previous = FrameSampler(3), None
            for t, frame in iter_frames(column[row].as_py()):
                if not sampler.keep(t):
                    continue
                thumb = thumbnail(frame_to_bgr(frame))
                if previous is not None:
                    diffs.append(float(np.abs(thumb - previous).mean()))
                previous = thumb

    diffs = np.array(diffs)
    print(f"{n_videos} videos, {len(diffs)} sampled frames compared with the previous one")
    for threshold in THRESHOLDS:
        print(f"  difference < {threshold:>3} grey levels: {np.mean(diffs < threshold):6.1%} of frames")


if __name__ == "__main__":
    main()
