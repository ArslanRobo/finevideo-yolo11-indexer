# Hardware

The full 50 files were run on my own workstation (a shared lab machine).

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB (driver 595.71.05) |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads |
| RAM | 121 GB |
| Disk | Samsung 990 EVO 2 TB NVMe SSD |
| OS | Ubuntu 25.10, kernel 6.17 |
| Python | 3.11 (conda env), torch 2.8.0+cu128, TensorRT 10.13.3.9 |

**Note: the GPU was shared.** Other services on the machine use it: a vLLM
engine (holding about 20 GB of memory, with bursts of compute when it serves
requests) and a live 4-camera detection pipeline with its Triton server (a
steady 15-20% of the GPU's compute). So these times are slower and noisier
than they would be on a dedicated GPU.

## Full run times (all 50 files: 1,650 videos, 128.3 h of video)

| version | wall time, start to finish | GPU conditions |
|---|---|---|
| First working version (PyTorch fp16, 3 frames/s) | 3,603 s (60.1 min) | shared |
| TensorRT FP16 + batched NMS + GPU thread | 1,014 s (16.9 min) | shared |
| **Final version** (as above, preprocessing built into the engine) | **1,127 s (18.8 min)** | live pipeline paused; vLLM serving requests |

`results.jsonl.gz` is the output of the final version's run. An earlier run
of the same final version, with every service active, took 1,332 s and gave
identical results.

For the final run the live camera pipeline was paused, but vLLM was serving
requests for much of it (about 1,300 prompt tokens/s at the start), so that
run still shared the GPU. Throughput moved between about 1,130 and 1,460
frames/s as vLLM's load changed. The previous version's 1,014 s run happened
to fall in a quieter period. On the 66-video subset the two versions took
37.5 s and 37.7 s: the same speed.

Each time is for `python run.py --input <folder> --output results.jsonl`
with the default settings (16 decode workers, batch 64, 3 frames/s), after
`python prepare.py` had built the TensorRT engine (an untimed one-off step of
5-9 minutes).
