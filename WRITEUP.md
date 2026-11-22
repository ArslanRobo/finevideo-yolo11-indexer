## 1 

The first working version on the full 50 files took 3,603 s (60.1 min) while the
final version took 1,127 s (18.8 min). Building the TensorRT engine with fp16
optimization saved me the most time.

## 2

Each Parquet file holds a few dozen videos, one per row, each a raw mp4. I read
that column once per file it's a single row group, so pulling one row means
reading the lot anyway then hand them to a pool of workers.
Each worker decodes its video right through, because I need real timestamps to
know which second a frame belongs to, but only turns three frames a second into
actual images. Those get resized to the model's input and sent back.
A background thread gathers them into batches of 64 and runs yolo11l a tensorRT
engine built up front, which does its own pixel prep. One nms pass gives the
counts per frame.Once a video's frames are all back, I apply the rules: an object counts if it
showed up in half that secondss frames, people is the median. That's one json
line per second.

## 3 
The model. In a batch of 64 frames the TensorRT engine takes about 27 ms of
roughly 31 ms, so call it 88%. Copying frames to the GPU is 3.4 ms, NMS and
counting 0.3 ms, copying results back is nothing.
I got those numbers with CUDA events around each step, median of 50 runs on a
real batch.

## 4 
Before touching speed I built a reference the slow way, plain ultralytics on
every frame of 66 videos, about four hours of footage. Every version since gets
scored against it on the brief's three numbers.The first two changes moved the score by 0.0001, which is noise. The batched NMS and the threading changed nothing at all. The output files were byte identical, which is the check I trust most. I also compared the new batch nms against
ultralytics own on 512 real frames: no differences.

## 5 

One bad file should not be able to stall the run, and the fix I got was if a
worker has been on the same video for ten minutes, kill it and start a fresh
one.sounded reasonable, so I built it.
The test for it hung. A worker killed mid-send leaves the shared queue's write
lock held, so every other worker blocks behind it. The fix caused exactly the
hang it was meant to prevent. What I ended up with has two levels: a worker gives up on its own video between frames, and only one that's frozen inside the decoder, and therefore holding
nothing, gets killed.

## 6
first, run everything on a GPU nobody else is using. Every number here was
measured next to a vLLM server and a live camera pipeline.
After that , widen the accuracy evidence. The 3 frames a second choice that I made rests on
66 videos from two files, I'd want more files and more variety behind it.
