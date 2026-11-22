"""Index FineVideo Parquet files: for each second of each video, which of
person / car / dog / laptop / cell phone appear, and how many people.

    python run.py --input <folder of .parquet files> --output results.jsonl

Run `python prepare.py` once beforehand (needs internet) to fetch yolo11l.pt.
The pipeline itself lives in videoindex/; see videoindex/pipeline.py.
"""

import argparse
import os

# The timed run has no internet: stop Ultralytics from trying to reach it.
os.environ.setdefault("YOLO_OFFLINE", "1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="folder holding the .parquet files")
    parser.add_argument("--output", required=True, help="JSON Lines file to write")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="decode processes (default: half the CPU threads)")
    parser.add_argument("--batch-size", type=int, default=64, help="frames per GPU batch")
    parser.add_argument("--sample-rate", type=float, default=3,
                        help="frames per second of video to run the model on; 0 = every frame")
    parser.add_argument("--backend", choices=["auto", "tensorrt", "pytorch"], default="auto",
                        help="auto = the TensorRT engine from prepare.py if present, else PyTorch")
    parser.add_argument("--fp32", action="store_true", help="PyTorch backend only: float32 instead of float16")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    from videoindex.pipeline import run  # after arg parsing so --help is instant

    run(
        input_dir=args.input,
        output_path=args.output,
        workers=args.workers,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate or None,
        device=args.device,
        half=not args.fp32,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
