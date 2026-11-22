"""One-time preparation, run before the timed run (it needs internet).

    python prepare.py

1. Downloads the official yolo11l.pt next to run.py.
2. Builds the TensorRT FP16 engine yolo11l.engine from it (a few minutes),
   with the input preprocessing built in (see videoindex/trt_engine.py).
   An engine only works on the GPU model that built it, so this has to run on
   the machine that will run run.py.
3. Checks that run.py's model loads.

If the engine build fails, run.py still works: it falls back to PyTorch,
just more slowly.
"""

import shutil
import sys
import time
from pathlib import Path

from videoindex.config import ENGINE, WEIGHTS


def download_weights():
    if WEIGHTS.exists():
        return
    from ultralytics.utils.downloads import attempt_download_asset

    # Downloads from the official Ultralytics assets release into the
    # current directory; move it next to run.py.
    downloaded = Path(attempt_download_asset("yolo11l.pt"))
    if downloaded.resolve() != WEIGHTS.resolve():
        shutil.move(str(downloaded), WEIGHTS)


def build_tensorrt_engine():
    from videoindex.trt_engine import add_preprocessing, build_engine, export_onnx

    start = time.perf_counter()
    onnx_path = Path(export_onnx(WEIGHTS))
    preprocessed = add_preprocessing(onnx_path, onnx_path.with_name(onnx_path.stem + "_preprocessed.onnx"))
    print(f"ONNX with preprocessing: {preprocessed}")
    build_engine(preprocessed, ENGINE)
    print(f"TensorRT engine: {ENGINE} ({ENGINE.stat().st_size / 1e6:.1f} MB, "
          f"built in {time.perf_counter() - start:.0f}s)")


def main():
    download_weights()
    print(f"weights: {WEIGHTS} ({WEIGHTS.stat().st_size / 1e6:.1f} MB)")

    if "--no-engine" in sys.argv:
        print("skipping the TensorRT engine; run.py will use PyTorch")
    else:
        try:
            build_tensorrt_engine()
        except Exception as exc:
            print(f"WARNING: TensorRT engine build failed ({type(exc).__name__}: {exc}). "
                  f"run.py will fall back to PyTorch.")

    from videoindex.detector import Detector

    detector = Detector()  # fails here, not in the timed run, if CUDA or the model is broken
    print(f"model loads on the GPU: OK (backend: {detector.backend})")


if __name__ == "__main__":
    main()
