"""Where does the time go inside one GPU batch?

    python tools/profile_batch.py <a .parquet file> [--batch 64] [--repeats 50]

Takes `batch` real letterboxed frames and times each step the GPU thread
does for a batch, with CUDA events (GPU time, not Python time):

  copy to GPU -> engine (preprocessing + YOLO11l) -> NMS + counting -> copy back

Reports the median over `repeats` batches, so short spikes from other jobs
on a shared GPU do not distort it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from videoindex.decode import frame_to_bgr, iter_frames  # noqa: E402
from videoindex.detector import Detector, batched_nms_counts, make_letterbox  # noqa: E402


def real_frames(parquet_path, n):
    letterbox = make_letterbox()
    column = pq.read_table(parquet_path, columns=["mp4"]).column("mp4")
    images = []
    for row in range(len(column)):
        for i, (_, frame) in enumerate(iter_frames(column[row].as_py())):
            if i % 10 == 0:
                image = letterbox(image=frame_to_bgr(frame))
                if not images or image.shape == images[0].shape:
                    images.append(image)
            if len(images) == n:
                return torch.from_numpy(np.stack(images))
    raise SystemExit("not enough frames of one size")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    detector = Detector()
    frames = real_frames(args.parquet, args.batch)
    print(f"backend {detector.backend}, batch {tuple(frames.shape)} uint8")

    steps = ["copy to GPU", "engine (preprocess + YOLO11l)", "NMS + counting", "copy back"]
    times = {step: [] for step in steps}
    with torch.cuda.stream(detector.stream):
        for repeat in range(args.repeats + 5):
            events = [torch.cuda.Event(enable_timing=True) for _ in range(len(steps) + 1)]
            events[0].record()
            on_gpu = frames.to(detector.device, non_blocking=True)
            events[1].record()
            predictions = detector.model(on_gpu)
            events[2].record()
            counts = batched_nms_counts(predictions, detector._column_of_class)
            events[3].record()
            counts.cpu()
            events[4].record()
            torch.cuda.synchronize()
            if repeat >= 5:  # skip warm-up
                for i, step in enumerate(steps):
                    times[step].append(events[i].elapsed_time(events[i + 1]))

    total = sum(np.median(t) for t in times.values())
    for step, t in times.items():
        median = np.median(t)
        print(f"  {step:32s} {median:6.2f} ms  ({median / total:5.1%})")
    print(f"  {'total':32s} {total:6.2f} ms  -> {args.batch * 1000 / total:.0f} frames/s")


if __name__ == "__main__":
    main()
