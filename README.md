# Video indexing at speed: FineVideo × YOLO11l

For every second of every video in a folder of FineVideo Parquet files, this
records which of **person, car, dog, laptop, cell phone** appear and how many
people are visible, using Ultralytics YOLO11l (`yolo11l.pt`, imgsz 640,
confidence 0.4, class-aware NMS at IoU 0.7).

## Install and run

Tested on Ubuntu with Python 3.11 and an NVIDIA driver that supports CUDA 12.
No CUDA toolkit is needed: PyTorch and TensorRT bring their own CUDA libraries
as pip packages.

```bash
# 1. System packages: OpenCV (installed by Ultralytics) needs libGL and glib
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0

# 2. Python packages (all 65 pinned to exact versions)
pip install -r requirements.txt

# 3. Preparation command: needs internet, run once before the timed run,
#    takes about 5-9 minutes. It
#      - downloads the official yolo11l.pt next to run.py,
#      - exports it to ONNX with the input preprocessing built in,
#      - builds a TensorRT FP16 engine (yolo11l.engine) for THIS GPU,
#      - checks that the model loads.
python prepare.py

# 4. The timed run (no internet needed)
python run.py --input <folder of .parquet files> --output results.jsonl
```

The engine must be built on the machine that runs `run.py`: a TensorRT engine
only works on the GPU model it was built on. If the engine build fails, or the
engine cannot be loaded, `run.py` says so on stderr and falls back to PyTorch
fp16 automatically. The results stay correct; the run is slower (about 2.4x).

Optional flags (the defaults are the tuned values):

| flag | default | meaning |
|---|---|---|
| `--workers` | half the CPU threads | decode processes |
| `--batch-size` | 64 | frames per GPU batch (the engine accepts up to 64) |
| `--sample-rate` | 3 | frames per second of video sent to the model (`0` = every frame) |
| `--backend` | `auto` | `auto` = the TensorRT engine if present, else PyTorch; or force `tensorrt` / `pytorch` |
| `--fp32` | off | PyTorch backend only: float32 instead of float16 |

Progress and timing go to stderr every 15 seconds. Results go only to the
`--output` file.

## How it works

```
reader thread ──video bytes──> decode workers ──frames──> GPU thread ──> results.jsonl
(main process)                 (N processes)              (main process)
```

1. **Reader** (`videoindex/parquet_source.py`) reads each Parquet file's
   `mp4` column once and queues the videos. A FineVideo file is a single row
   group, so reading per row would re-read the whole file each time.
2. **Decode workers** (`videoindex/pipeline.py: decode_worker`) decode every
   frame with PyAV, because exact timestamps decide which second a frame
   belongs to, but convert only a sample of frames to images: the first frame
   in each 1/3-second slot. Frames with a rotation tag are turned upright.
   Each sampled frame is letterboxed with Ultralytics' own `LetterBox`, so the
   model gets exactly the input `predict()` would build (640x360 becomes
   640x384). Frames travel to the main process through shared memory.
3. **GPU thread** gets batches of same-size frames and runs YOLO11l as a
   **TensorRT FP16 engine** (`videoindex/trt_engine.py`). The engine takes the
   letterboxed uint8 BGR frames as they are: the uint8 to float conversion,
   HWC to CHW, BGR to RGB and /255 are nodes inside the ONNX graph, and so
   inside the engine. NMS then runs for the whole batch in one GPU call
   (`videoindex/detector.py: batched_nms_counts`), giving the same counts as
   Ultralytics' `non_max_suppression`. The main thread builds the next batch
   while the GPU works.
4. **Per-second rules** (`videoindex/seconds.py`): an object is present if it
   was found in at least half of the second's sampled frames; the people count
   is the median, rounded down at a half. These are the brief's section 5
   rules, applied to the sampled frames.

Failures: a video that cannot be decoded gets one error line; a video that
fails partway keeps the seconds already decoded plus an error line; if a
decode process crashes, its video is reported and a new process replaces it.
A video that takes over 10 minutes to decode is given up (error line), and a
decoder frozen for 15 minutes is killed and replaced, so no single file can
stall the run.
A video's timestamps start at its first decoded frame, as the brief defines.

## Why 3 frames per second is enough

The graders' reference runs the model on every frame. Running it on 3
evenly spread frames per second instead, and applying the same rules, was
measured against an every-frame reference built with plain Ultralytics
`predict()` (`tools/reference.py`) on 66 videos (4.3 hours):

| frames/s | objects F1 | people exact | people within 1 |
|---|---|---|---|
| 1 | 0.956 | 0.872 | 0.978 |
| **3 (default)** | **0.983** | **0.940** | **0.996** |
| 5 | 0.990 | 0.959 | 0.999 |
| thresholds | 0.95 | 0.85 | 0.95 |

Resampling the 66 videos into 200 random test sets of 13-hour size, 3 frames/s
still gives F1 ≥ 0.980 and exact ≥ 0.932 in 95% of them. An odd number of
samples also avoids a tie bias: with 2 samples, 1 of 2 already counts as
"at least half". Details in `NOTES.md`.

## Checking correctness

```bash
python tools/test_rules.py                # per-second rules and frame sampling
python tools/test_rotation.py             # rotation matches ffmpeg (needs: sudo apt-get install -y ffmpeg)
python tools/test_nms.py <file.parquet>   # batched NMS == Ultralytics NMS, on real frames
python tools/reference.py --input <folder> --output reference.jsonl   # every frame, plain predict()
python tools/compare.py results.jsonl reference.jsonl                 # the brief's three metrics
python tools/test_robustness.py <small folder>   # a frozen decoder or failing reader cannot hang the run
```

Measurement tools: `tools/profile_batch.py` (GPU time per step of a batch),
`tools/bench_decode.py` (CPU decode throughput alone),
`tools/measure_duplicates.py` (how many sampled frames are near-copies).

Only `tools/test_rotation.py` needs the `ffmpeg` command; `run.py` does not
(PyAV ships its own FFmpeg libraries).

## Files

| path | what |
|---|---|
| `run.py` | entry point |
| `prepare.py` | one-time download of `yolo11l.pt` and TensorRT engine build |
| `videoindex/` | the pipeline: reading, decoding, detection, per-second rules |
| `tools/` | reference, scorer, tests and benchmarks |
| `eval/` | the every-frame reference for the 66 evaluation videos, and the sweep script |
| `results.jsonl.gz` | results for the 50 FineVideo files |
| `HARDWARE.md` | where the full run was done and how long it took |
| `NOTES.md` | dated log of every measurement and change |
| `WRITEUP.md` | answers to the write-up questions |
| `ai_chats/` | AI chat history |
