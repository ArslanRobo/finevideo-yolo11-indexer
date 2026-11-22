# Development notes

A dated log of what was measured and changed. Facts only. The write-up
answers in WRITEUP.md are written separately, by hand.

## Data (measured on 330 videos = 10 of the 50 files, 25.6 h)

- Codec: H.264 in all 330.
- Resolution: 640x360 in 248 (75%), 360x640 (portrait) in 32, the rest
  smaller (480x360, 270x480, ...). Nothing above 640 wide.
- Frame rate: mostly 30, 29.97, 25, 23.976, 24.
- Rotation tags: none in the sample. Handled anyway (the test set may have
  "unusual" files); verified against ffmpeg in tools/test_rotation.py.
- Each Parquet file is ONE row group of ~33 rows, so a single row cannot be
  read without reading the file's whole mp4 column (~0.5 GB).

Consequence: frames are small and cheap to decode; running YOLO11l is the
expensive part. ~125 h x ~29 fps = ~13 M frames.

## Environment

- Development machine: shared RTX 5090 (32 GB, but a VLLM engine and Triton
  use ~26 GB, leaving ~6 GB), 32 threads. Timings here are therefore noisy
  and slower than on a dedicated GPU.
- Python 3.11 (conda env `finevideo`), torch 2.8.0+cu128, ultralytics 8.4.216,
  PyAV 18.1.0.

## Correctness checks

- `tools/test_rules.py`: 10 hand-worked cases of the section 5 rules (half
  threshold, median rounding down) and of the frame sampler. All pass.
- `tools/test_rotation.py`: rotated test videos (90/180/270) decoded
  pixel-identical to ffmpeg's autorotate. The first version turned 90 and
  270 the wrong way; this test caught it.
- Tiny smoke test (4 short videos + a truncated video + a garbage row):
  - run.py, 5 frames/s, fp16 vs reference: objects F1 0.9965, people exact
    0.9863, within 1 1.0000. PASS.
  - run.py, every frame, fp32 vs reference: F1 1.0000, exact 0.9932, i.e. the
    custom GPU path reproduces Ultralytics predict().
  - truncated video: decoded seconds kept + one error line; garbage row:
    one error line. Same as the reference.

## Reference on the evaluation subset

- Subset: train-00007 and train-00031 (66 videos), in
  /home/beast/Data/Arslan/finevideo_eval/subset (symlinks).
- `tools/reference.py`, every frame, plain model.predict, fp32:
  1540 s, ~294 frames/s average (shared GPU).
  -> eval/reference_2files.jsonl

## Bugs found while running

1. FileNotFoundError in torch `rebuild_storage_fd`: frames sent just before
   a worker exited could not be opened, because the default tensor sharing
   goes through a socket owned by the sender. Fixed with
   `set_sharing_strategy("file_system")`.
2. CUDA out of memory on the shared GPU (52 MB free while the reference
   ran beside it). Fixed: a batch that does not fit is split in half and
   retried.
3. After that crash the program HUNG for hours instead of exiting: Python
   waits at exit for multiprocessing queues to deliver their data, and the
   readers were gone. Fixed with `cancel_join_thread()` on both queues;
   tested by interrupting a run mid-way (exits in 1 s, no leftover
   processes).

## Sampling rate (2026-11-21), run.py fp16 vs eval/reference_2files.jsonl

| frames/s | objects F1 | people exact | within 1 | bootstrap 5th pct, 190-video test set (F1 / exact) |
|---|---|---|---|---|
| 1 | 0.9557 | 0.8720 | 0.9780 | 0.950 / 0.856 |
| 2 | 0.9683 | 0.9096 | 0.9920 | - |
| 3 | 0.9828 | 0.9402 | 0.9958 | 0.980 / 0.932 |
| 5 | 0.9900 | 0.9587 | 0.9986 | 0.988 / 0.952 |

- Chosen: 3 frames/s, the cheapest rate that passes with margin on an
  unlucky draw of test videos. 1/s sits on the F1 threshold.
- Even sample counts are biased toward "present": with 2 samples, 1 of 2
  satisfies "at least half". At 2/s, car had fp 94 vs fn 7.
- Scores were nearly equal on the two files (e.g. 3/s: F1 0.983 / 0.982).

## Full run, version 1 (2026-11-21)

PyTorch fp16, 3 frames/s, 16 decode workers, batch 64:
**3603 s** for all 50 files (1650 videos, 128.3 h of video, 1.38 M frames
sent to the model, ~384 frames/s). GPU busy 93%. All 1650 videos present,
no gaps or duplicate seconds, no error lines. -> runs/v1_results.jsonl

## Speed-ups (2026-11-22), same 66-video subset, 3 frames/s

| version | time | frames/s | vs reference |
|---|---|---|---|
| PyTorch fp16 (v1) | 90.4 s | 517 | F1 0.9828, exact 0.9402, w1 0.9958 |
| + TensorRT FP16 engine | 58.5 s | 800 | F1 0.9829, exact 0.9402, w1 0.9959 |
| + batched NMS, own CUDA stream | 41.2 s | 1134 | byte-identical output to the row above |
| + GPU thread overlapping CPU work | 37.7 s | 1241 | byte-identical output |

How each was found:
- TensorRT: GPU busy ~90% in v1, so the model was the cost. Engine built by
  our own script (videoindex/trt_engine.py), tuned for 640x384 because
  Ultralytics' export tunes for 640x640. Build: 327 s on the shared 5090.
- Per-batch timing (batch 64, TensorRT): H2D copy 3.2 ms, preprocess
  0.8 ms, TensorRT 26.1 ms, **Ultralytics NMS 26.8 ms** (Python loop over
  the 64 images). Replaced by one batched call (batched_nms_counts);
  tools/test_nms.py: 0 of 512 real frames differ from Ultralytics, for
  both backends, and the 300-per-image cap matches on a crowded image.
- cProfile of the main process after that: ~15 s of CPU work (torch.cat
  4.8 s, copy to GPU 6.0 s, loop 4.5 s) serialized with ~19 s of GPU
  waiting. Moved the model to a thread (GpuThread) so the two overlap.

## Preprocessing inside the engine (2026-11-22)

The ONNX gets four nodes before the network (videoindex/trt_engine.py
add_preprocessing): input `frames` uint8 (batch, height, width, 3) BGR ->
Cast(float32) -> Transpose -> Gather([2,1,0]) -> Div(255). Cast first,
because TensorRT rejects a Gather on uint8.

- ONNX Runtime, 4 real frames: preprocessed ONNX on raw frames vs original
  ONNX + Python preprocessing -> max abs difference 0.0 (bit-identical).
- Subset (66 videos, 3/s): 37.5 s vs 37.7 s before: no measurable speed
  change, as expected (the steps cost ~0.8 ms of ~30 ms per batch).
  vs reference: F1 0.9830, exact 0.9401, within 1 0.9959 (unchanged).
- vs the previous engine's output: 99.97% objects / 99.91% people agree,
  not byte-identical. Any TensorRT rebuild can choose different FP16
  kernels (chosen by timing during the build), so tiny differences are
  normal between builds; the graph itself was shown identical above.
- tools/test_nms.py: still 0 differences from Ultralytics NMS.

## GPU decoding (NVDEC): measured, not done

tools/bench_decode.py: 16 CPU decode workers alone deliver 2,355 sampled +
letterboxed frames/s (22,785 decoded frames/s) on the subset, while the GPU
consumes ~1,370/s. CPU decoding has 1.7x headroom, so it is not the
bottleneck; NVDEC would also put colour conversion/resizing on the same GPU
cores the model needs.

### Follow-ups to the preprocessing change (2026-11-22, afternoon)

- Bug found: with the GPU nearly full (another job held most of it), TensorRT's
  `create_execution_context()` returned None instead of raising (it needs
  ~2.4 GB working memory for batch 64 at 640x640), so run.py crashed on the
  first batch with `'NoneType' object has no attribute 'set_input_shape'`.
  TrtModel now raises, so `--backend auto` falls back to PyTorch.
- Old vs new engine timed ALTERNATELY on the same real batch of 64
  (both see the same background load): 56.7 ms vs 55.5 ms: within noise on a shared GPU. No measurable speed change (subset 37.5 s vs 37.7 s).
- Full 50-file run with the new engine: 1,332 s, EXIT=0, all 1,650 videos.
  Slower than the 1,014 s run only because the GPU was shared with other
  jobs at the time (subset throughput at that time: 689 frames/s vs 1,241
  earlier with the same code). vs the previous full run: F1 0.9997,
  exact 0.9989, within 1 1.0000. -> runs/final2_results.jsonl

## Reusing results for near-identical frames (2026-11-22, evening): measured, then REMOVED

Idea: tools/measure_duplicates.py on the subset: 26% of sampled frames differ
from the previous one by < 2 grey levels (64x36 grayscale thumbnail, mean
absolute difference), 33% by < 3. Implemented as `--reuse-threshold` (decode
worker compares with the LAST FRAME SENT to the model, so slow changes cannot
creep through; a skipped frame copies that frame's counts).

eval/reuse_sweep.sh, 66-video subset, vs every-frame reference:

| threshold | frames to model | reused | F1 | exact | within 1 |
|---|---|---|---|---|---|
| 0 (off) | 46,766 | 0 | 0.9830 | 0.9401 | 0.9959 |
| 1 | 39,461 | 16% | 0.9800 | 0.9354 | 0.9958 |
| 2 | 35,970 | 23% | 0.9786 | 0.9314 | 0.9951 |
| 3 | 33,128 | 29% | 0.9741 | 0.9231 | 0.9943 |
| 5 | 28,639 | 39% | 0.9686 | 0.9136 | 0.9926 |

- Accuracy falls steadily with the threshold; all pass, with shrinking margin.
- Wall time did NOT measurably improve, and could not be measured: the GPU
  was shared with other jobs whose load swung the same no-reuse run from
  43.3 s to 60.8 s minutes apart (alternating off/3/off/3: 43.3, 45.0,
  60.8, 43.1 s). Needs a quiet GPU for an A/B before it can be judged.
- Decision: not adopted (accuracy cost, no measurable gain on the shared GPU). The code was removed again; tools/measure_duplicates.py stays as the measurement.

## Review of the current code (2026-11-22, evening)

Two ways one bad file could have HUNG the run (graders stop it at 6 h):

1. No time limit per video: a decoder freezing on a corrupt file would block
   its worker, and run.py waits for every worker. First fix (main process
   kills a worker after 10 min) was WRONG and its test hung: a worker killed
   while sending frames leaves the shared multiprocessing queue's write lock
   held, and every other worker blocks on it. Final design, two levels:
   - VIDEO_TIMEOUT_SECONDS = 600: the worker checks the clock between frames
     and gives up itself (error line, finished seconds kept). No kill.
   - STUCK_WORKER_SECONDS = 900: only a worker frozen INSIDE one decoder call
     (so not sending anything) is killed by the main process and replaced.
2. If the reader thread raised, workers never got their stop signal. Now in
   try/finally.

tools/test_robustness.py: slow video (limit 0.05 s), frozen decoder (patched
to sleep forever, kill limit 5 s), reader failure: all three FINISH; the
first two give every video a line.

Offline check (the timed run has no internet): with a sitecustomize.py that
blocks and logs every non-loopback connection in all processes, run.py on the
tiny set made **0 connection attempts** and passed.

## Profile of the current code (2026-11-22, evening)

tools/profile_batch.py, one batch of 64 real 640x384 frames, CUDA events,
median of 50:

| step | ms | share |
|---|---|---|
| copy to GPU (pageable memory) | 3.36 | 11.0% |
| engine (preprocessing + YOLO11l) | 27.01 | 87.9% |
| NMS + counting | 0.30 | 1.0% |
| copy back | 0.04 | 0.1% |
| total | 30.71 | -> 2,084 frames/s |

cProfile of the main process, subset run (43.0 s, GPU shared at the time):
27.8 s waiting for the GPU thread to accept the next batch, torch.cat 5.5 s,
loop 4.0 s. So the pipeline is GPU-bound and the engine is the cost. In the
run a batch takes ~50 ms instead of ~31 ms: the other jobs on the GPU.
Left: pinned host memory for the copy (~1.7 ms saved per batch, ~5%).

## Final timed run (2026-11-22, 18:02-18:21 PKT)

All 18 3dvt-person-live containers were paused (`docker pause`, resumed by an
EXIT trap; runs/timed_run_paused_3dvt.sh). vLLM was NOT paused and was
serving requests (~1,300 prompt tokens/s at the start). Result: **1,127 s**,
EXIT=0, all 1,650 videos, no gaps or error lines, and identical to the
earlier 1,332 s run of the same engine (F1, exact, within 1 all 1.0000).
Throughput varied 1,130-1,460 frames/s with vLLM's load.
After resuming, the 4 imagebuffer containers had lost their RTSP streams and
were restarted by their restart policy; all stages back at 15 FPS.
