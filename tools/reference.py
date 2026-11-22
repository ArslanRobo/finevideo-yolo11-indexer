"""Slow reference: run YOLO11l on EVERY frame, the way the graders describe.

    python tools/reference.py --input <folder of .parquet> --output reference.jsonl

This is deliberately the plainest possible version, sharing as little as
possible with the fast pipeline so that it can catch the fast pipeline's
mistakes:

* one video at a time, every frame decoded and sent to the model;
* Ultralytics' own `model.predict` with its defaults (float32, letterbox,
  NMS IoU 0.7), only conf=0.4 and imgsz=640 set, as section 3 says;
* the section 5 rules: present if found in >= half the frames, median
  people count rounded down.

What it does share with run.py: decode.iter_frames / frame_to_bgr (how
frames and timestamps are read) and seconds.video_lines (the per-second
rules). Those are checked separately by the tests in tools/test_rules.py.

It doubles as the "first working version" (write-up question 1).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from videoindex.config import CLASS_IDS, CONFIDENCE, IMAGE_SIZE, TARGET_CLASSES, WEIGHTS  # noqa: E402
from videoindex.decode import frame_to_bgr, iter_frames, second_of  # noqa: E402
from videoindex.parquet_source import iter_videos, list_parquet_files  # noqa: E402
from videoindex.seconds import error_line, video_lines  # noqa: E402

BATCH = 32


def count_frames(model, images):
    """Per-frame counts (n, 5) for a list of BGR images, via model.predict."""
    results = model.predict(images, imgsz=IMAGE_SIZE, conf=CONFIDENCE, verbose=False)
    rows = []
    for r in results:
        classes = r.boxes.cls.int().tolist()
        rows.append([classes.count(class_id) for class_id in CLASS_IDS])
    return rows


def process_video(model, video_id, mp4_bytes):
    seconds, counts, batch, batch_seconds = [], [], [], []
    n_seconds, error = 0, None
    try:
        for t, frame in iter_frames(mp4_bytes):
            s = second_of(t)
            n_seconds = max(n_seconds, s + 1)
            batch.append(frame_to_bgr(frame))
            batch_seconds.append(s)
            if len(batch) == BATCH:
                counts += count_frames(model, batch)
                seconds += batch_seconds
                batch, batch_seconds = [], []
        if n_seconds == 0:
            error = "no frames could be decoded"
    except Exception as exc:
        error = f"could not decode video: {type(exc).__name__}"
    if batch:
        counts += count_frames(model, batch)
        seconds += batch_seconds

    lines = video_lines(video_id, seconds, np.asarray(counts).reshape(-1, len(TARGET_CLASSES)), n_seconds)
    if error:
        lines.append(error_line(video_id, error))
    return lines, len(seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--files", type=int, default=None, help="only the first N parquet files")
    parser.add_argument("--skip-files", type=int, default=0, help="skip the first N parquet files")
    args = parser.parse_args()

    model = YOLO(str(WEIGHTS))
    files = list_parquet_files(args.input)[args.skip_files:]
    if args.files:
        files = files[:args.files]

    start = time.perf_counter()
    n_frames = 0
    with open(args.output, "w") as out:
        for path in files:
            for video_id, mp4_bytes, error in iter_videos(path):
                if error:
                    lines, n = [error_line(video_id, error)], 0
                else:
                    lines, n = process_video(model, video_id, mp4_bytes)
                for line in lines:
                    out.write(json.dumps(line) + "\n")
                out.flush()
                n_frames += n
                elapsed = time.perf_counter() - start
                print(f"[{elapsed:7.0f}s] {video_id}: {n} frames, total {n_frames / elapsed:.0f} frames/s",
                      file=sys.stderr, flush=True)
    print(f"Done in {time.perf_counter() - start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
