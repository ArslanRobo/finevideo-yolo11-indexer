"""How fast can the CPU decode workers produce frames, with no GPU involved?

    python tools/bench_decode.py <folder of .parquet> [--workers 16]

Does the same work as a decode worker in run.py (decode every frame, keep
3 per second, convert and letterbox those) in N processes, and reports
frames per second. If this is well above the rate the GPU consumes
frames, decoding is not the bottleneck and moving it to the GPU (NVDEC)
cannot make the run faster.
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def decode_one(mp4_bytes):
    import cv2

    from videoindex.decode import FrameSampler, frame_to_bgr, iter_frames
    from videoindex.detector import make_letterbox

    cv2.setNumThreads(1)
    letterbox, sampler = make_letterbox(), FrameSampler(3)
    decoded = sampled = 0
    duration = 0.0
    for t, frame in iter_frames(mp4_bytes):
        decoded += 1
        duration = t
        if sampler.keep(t):
            letterbox(image=frame_to_bgr(frame))
            sampled += 1
    return decoded, sampled, duration


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    videos = []
    for path in sorted(Path(args.input).glob("*.parquet")):
        column = pq.read_table(path, columns=["mp4"]).column("mp4")
        videos += [column[i].as_py() for i in range(len(column))]

    with mp.get_context("spawn").Pool(args.workers) as pool:
        pool.map(decode_one, videos[: args.workers])  # warm up: imports in every process
        start = time.perf_counter()
        results = pool.map(decode_one, videos, chunksize=1)
        elapsed = time.perf_counter() - start

    decoded, sampled, duration = map(sum, zip(*results))
    print(f"{args.workers} workers: {elapsed:.1f}s for {len(videos)} videos ({duration / 3600:.2f} h of video)")
    print(f"  decoded              {decoded / elapsed:8.0f} frames/s")
    print(f"  sampled + letterboxed {sampled / elapsed:7.0f} frames/s   <- what the GPU loop receives")


if __name__ == "__main__":
    main()
