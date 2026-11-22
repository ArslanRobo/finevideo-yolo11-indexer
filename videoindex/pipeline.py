"""The whole run: Parquet files in, results.jsonl out.

    reader thread ──(video bytes)──> decode workers ──(frames)──> GPU loop ──> results.jsonl
    (main process)                   (N processes)                (main process)

* Reader thread: reads each Parquet file once and queues its videos.
* Decode workers: separate processes, so decoding runs on many CPU cores at
  once. Each decodes a whole video, samples a few frames per second,
  letterboxes them and sends them on in chunks. Frames travel as torch
  tensors, which torch.multiprocessing moves through shared memory instead
  of copying them through a pipe.
* GPU loop: collects frames of the same size into batches, runs YOLO11l,
  and once every sampled frame of a video is counted, writes that video's
  lines.

Failures never stop the run: a video that cannot be decoded gets an error
line, a video that fails partway keeps its finished seconds plus an error
line, and if a decode process dies, its video is reported and a replacement
process is started.
"""

import json
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp

from .decode import FrameSampler, frame_to_bgr, iter_frames, second_of
from .parquet_source import iter_videos, list_parquet_files
from .seconds import error_line, video_lines

# How frame tensors are shared between processes. The default
# ("file_descriptor") hands each tensor over through a socket owned by the
# SENDING process, so frames sent just before a worker exits can no longer
# be opened by the GPU loop (FileNotFoundError in rebuild_storage_fd).
# "file_system" uses named shared-memory files that outlive the sender.
# Must be set in every process, so it lives at import time of this module.
mp.set_sharing_strategy("file_system")

# Time limits, so one bad file cannot stall the whole run. The longest
# FineVideo video seen (11 min of footage) decodes in about 15 s, so both are
# very generous.
#
# VIDEO_TIMEOUT_SECONDS: a worker checks the clock between frames and gives
#   up on a video that takes longer (error line, finished seconds kept).
# STUCK_WORKER_SECONDS: if a worker is STILL on the same video after this,
#   it is frozen inside one decoder call and cannot check the clock, so the
#   main process kills it and starts a new one. Killing is the last resort
#   because a worker killed while sending frames would leave the shared
#   results queue locked and hang every other worker; a frozen worker is not
#   sending anything, which is why the kill only happens on this second,
#   longer limit.
VIDEO_TIMEOUT_SECONDS = 600
STUCK_WORKER_SECONDS = 900

# Messages from workers to the GPU loop are tuples whose first item says what
# they are:
#   ("start",  worker_id, video_id)
#   ("frames", video_id, seconds_tensor, frames_tensor)
#   ("done",   worker_id, video_id, n_frames_sent, n_seconds, error_or_None)
#   ("exit",   worker_id)


# ---------------------------------------------------------------------------
# Decode worker (runs in its own process)
# ---------------------------------------------------------------------------

def decode_worker(worker_id, tasks, results, sample_rate, chunk_size, decoder_threads):
    # Each process gets one core's worth of work; stop libraries from
    # starting their own thread pools on top of that.
    import cv2

    from .detector import make_letterbox

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    letterbox = make_letterbox()

    while True:
        task = tasks.get()
        if task is None:  # no more videos
            break
        video_id, mp4_bytes = task
        results.put(("start", worker_id, video_id))

        sampler = FrameSampler(sample_rate)
        chunk_seconds, chunk_frames = [], []
        frames_sent = 0
        n_seconds = 0
        error = None

        def send_chunk():
            nonlocal frames_sent, chunk_seconds, chunk_frames
            if chunk_frames:
                results.put((
                    "frames",
                    video_id,
                    torch.tensor(chunk_seconds, dtype=torch.int32),
                    torch.from_numpy(_stack(chunk_frames)),
                ))
                frames_sent += len(chunk_frames)
                chunk_seconds, chunk_frames = [], []

        started = time.monotonic()
        try:
            for t, frame in iter_frames(mp4_bytes, decoder_threads):
                if time.monotonic() - started > VIDEO_TIMEOUT_SECONDS:
                    error = f"gave up: decoding took longer than {VIDEO_TIMEOUT_SECONDS} s"
                    break
                second = second_of(t)
                n_seconds = max(n_seconds, second + 1)
                if not sampler.keep(t):
                    continue
                image = letterbox(image=frame_to_bgr(frame))
                # A chunk must hold frames of one size; flush if the size changes.
                if chunk_frames and image.shape != chunk_frames[0].shape:
                    send_chunk()
                chunk_seconds.append(second)
                chunk_frames.append(image)
                if len(chunk_frames) >= chunk_size:
                    send_chunk()
            if n_seconds == 0 and error is None:
                error = "no frames could be decoded"
        except Exception as exc:
            error = f"could not decode video: {type(exc).__name__}: {exc}"[:200]

        send_chunk()  # frames decoded before an error are still counted
        results.put(("done", worker_id, video_id, frames_sent, n_seconds, error))

    results.put(("exit", worker_id))


def _stack(images):
    import numpy as np

    return np.stack(images)


# ---------------------------------------------------------------------------
# Reader thread (runs in the main process)
# ---------------------------------------------------------------------------

def read_parquet_files(files, tasks, results, n_workers, dispatched):
    """Queue every video for the workers, then one stop signal per worker.

    Rows that cannot be read are reported straight to the GPU loop as
    finished videos with an error.
    """
    try:
        for path in files:
            for video_id, mp4_bytes, error in iter_videos(path):
                dispatched.append(video_id)
                if error:
                    results.put(("done", None, video_id, 0, 0, error))
                else:
                    tasks.put((video_id, mp4_bytes))
    except Exception as exc:
        # Should not happen (iter_videos reports bad files itself), but if it
        # does, the stop signals below must still go out, or every worker
        # would wait for more videos forever and the run would hang.
        print(f"reader stopped early: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        for _ in range(n_workers):
            tasks.put(None)


# ---------------------------------------------------------------------------
# GPU loop helpers
# ---------------------------------------------------------------------------

@dataclass
class VideoResult:
    """Everything known about one video while its frames are being counted."""

    frame_seconds: list = field(default_factory=list)
    frame_counts: list = field(default_factory=list)
    frames_counted: int = 0
    frames_expected: int | None = None  # known once the worker says "done"
    n_seconds: int = 0
    error: str | None = None
    crashed: bool = False  # its decode process died; no "done" will come

    def is_complete(self):
        return self.frames_expected is not None and self.frames_counted == self.frames_expected


class ShapeBatcher:
    """Groups incoming frames into batches of one image size.

    Frames from different videos can share a batch as long as their
    letterboxed size matches (most FineVideo videos are 640x360 -> 640x384).
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.pending = defaultdict(list)  # shape -> [(video_id, seconds, frames), ...]
        self.counts = defaultdict(int)    # shape -> number of pending frames

    def add(self, video_id, seconds, frames):
        shape = tuple(frames.shape[1:])
        self.pending[shape].append((video_id, seconds, frames))
        self.counts[shape] += len(seconds)

    def full_batches(self):
        for shape in list(self.pending):
            while self.counts[shape] >= self.batch_size:
                yield self._take(shape, self.batch_size)

    def all_batches(self):
        """Everything still pending, including part-filled batches."""
        for shape in list(self.pending):
            while self.counts[shape] > 0:
                yield self._take(shape, self.batch_size)

    def is_empty(self):
        return not any(self.counts.values())

    def _take(self, shape, n):
        """Remove up to n frames of one shape. Returns [(video_id, seconds, frames)]."""
        items = self.pending[shape]
        taken, got = [], 0
        while items and got < n:
            video_id, seconds, frames = items[0]
            k = min(n - got, len(seconds))
            taken.append((video_id, seconds[:k], frames[:k]))
            got += k
            if k == len(seconds):
                items.pop(0)
            else:
                items[0] = (video_id, seconds[k:], frames[k:])
        self.counts[shape] -= got
        if not items:
            del self.pending[shape], self.counts[shape]
        return taken


class GpuThread:
    """Runs the model on its own thread, so the main thread can receive
    frames and build the next batch while the GPU works on this one.

    Without it the two took turns: profiling showed ~15 s of the main
    thread's CPU work (joining frames, copying them to the GPU, bookkeeping)
    added on top of the GPU time for a 4.3 h subset. A thread is enough
    because PyTorch and TensorRT release Python's GIL while the GPU runs.

    submit() hands over a batch (it waits if `depth` batches are already
    queued); finished() returns the batches that are done.
    """

    def __init__(self, detector, depth=2):
        self.detector = detector
        self.inbox = queue.Queue(maxsize=depth)
        self.outbox = queue.Queue()
        self.in_flight = 0       # batches submitted but not yet collected
        self.busy_seconds = 0.0  # time spent in the model, for the progress line
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, items):
        frames = torch.cat([f for _, _, f in items])
        self.in_flight += 1
        self.inbox.put((items, frames))

    def finished(self, wait=False):
        """Yield (items, counts) for every completed batch. With wait=True,
        block until every submitted batch is done."""
        while self.in_flight:
            try:
                items, result = self.outbox.get(block=wait)
            except queue.Empty:
                return
            self.in_flight -= 1
            if isinstance(result, BaseException):
                raise result  # a GPU error stops the run rather than being lost
            yield items, result

    def close(self):
        self.inbox.put(None)

    def _loop(self):
        while (job := self.inbox.get()) is not None:
            items, frames = job
            t0 = time.perf_counter()
            try:
                counts = self.detector.count_objects(frames)
            except BaseException as exc:
                self.outbox.put((items, exc))
                continue
            self.busy_seconds += time.perf_counter() - t0
            self.outbox.put((items, counts))


class Stats:
    """Timing counters, printed while running and at the end."""

    def __init__(self):
        self.start = time.perf_counter()
        self.gpu_seconds = 0.0
        self.wait_seconds = 0.0
        self.frames = 0
        self.batches = 0
        self.videos_done = 0
        self.video_seconds = 0
        self._last_report = self.start

    def report(self, total_videos, force=False):
        now = time.perf_counter()
        if not force and now - self._last_report < 15:
            return
        self._last_report = now
        elapsed = now - self.start
        print(
            f"[{elapsed:7.1f}s] videos {self.videos_done}/{total_videos or '?'}"
            f" | {self.frames} frames ({self.frames / elapsed:.0f}/s)"
            f" | {self.video_seconds / 3600:.2f} h of video"
            f" | GPU busy {self.gpu_seconds / elapsed:.0%}, waiting for frames {self.wait_seconds / elapsed:.0%}"
            f" | avg batch {self.frames / max(self.batches, 1):.1f}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(input_dir, output_path, workers, batch_size, sample_rate, device, half, backend="auto",
        decoder_threads=1):
    from .detector import Detector  # imported here so workers do not load the model
    from .trt_engine import MAX_BATCH

    files = list_parquet_files(input_dir)
    detector = Detector(device=device, half=half, backend=backend)
    if detector.backend == "tensorrt" and batch_size > MAX_BATCH:
        batch_size = MAX_BATCH  # the engine was built for at most this many frames
    print(f"{len(files)} parquet files, {workers} decode workers, batch {batch_size}, "
          f"sampling {sample_rate or 'every'} frame(s)/s, model: {detector.backend}", file=sys.stderr)

    # "spawn" starts clean processes; forking a process that has already
    # touched CUDA is unsafe.
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue(maxsize=workers * 2)       # bounded: limits videos held in RAM
    results = ctx.Queue(maxsize=workers * 8)     # bounded: limits frames in flight
    chunk_size = max(1, batch_size // 2)

    def start_worker(worker_id):
        process = ctx.Process(
            target=decode_worker,
            args=(worker_id, tasks, results, sample_rate, chunk_size, decoder_threads),
            daemon=True,
        )
        process.start()
        return process

    processes = {i: start_worker(i) for i in range(workers)}
    next_worker_id = workers

    dispatched = []  # every video id handed out, to catch any that never report back
    reader = threading.Thread(
        target=read_parquet_files,
        args=(files, tasks, results, workers, dispatched),
        daemon=True,
    )
    reader.start()

    batcher = ShapeBatcher(batch_size)
    gpu = GpuThread(detector)
    videos = {}                  # video_id -> VideoResult, while in progress
    written = set()              # video ids whose lines are written
    current_video = {}           # worker_id -> (video id it is decoding, when it started)
    timed_out = set()            # worker ids stopped for taking too long
    stats = Stats()

    out = open(output_path, "w")

    def write_video(video_id, result):
        lines = []
        if result.n_seconds:
            lines = video_lines(video_id, result.frame_seconds, result.frame_counts, result.n_seconds)
        if result.error:
            lines.append(error_line(video_id, result.error))
        for line in lines:
            out.write(json.dumps(line) + "\n")
        written.add(video_id)
        stats.videos_done += 1
        stats.video_seconds += result.n_seconds

    def finish_if_complete(video_id):
        result = videos.get(video_id)
        if result is not None and result.is_complete():
            write_video(video_id, videos.pop(video_id))

    def apply_counts(items, counts):
        """Store a finished batch's per-frame counts with their videos."""
        stats.frames += len(counts)
        stats.batches += 1
        stats.gpu_seconds = gpu.busy_seconds
        offset = 0
        for video_id, seconds, chunk in items:
            n = len(seconds)
            result = videos[video_id]
            result.frame_seconds.extend(seconds.tolist())
            result.frame_counts.extend(counts[offset:offset + n].numpy())
            result.frames_counted += n
            offset += n
            finish_if_complete(video_id)

    def handle(message):
        kind = message[0]
        if kind == "start":
            _, worker_id, video_id = message
            current_video[worker_id] = (video_id, time.monotonic())
            videos.setdefault(video_id, VideoResult())
        elif kind == "frames":
            _, video_id, seconds, frames = message
            videos.setdefault(video_id, VideoResult())
            batcher.add(video_id, seconds, frames)
        elif kind == "done":
            _, worker_id, video_id, n_sent, n_seconds, error = message
            result = videos.setdefault(video_id, VideoResult())
            result.frames_expected = n_sent
            result.n_seconds = n_seconds
            result.error = error
            current_video.pop(worker_id, None)
            finish_if_complete(video_id)
        elif kind == "exit":
            processes.pop(message[1], None)

    def replace_crashed_workers():
        """A decoder can crash, or freeze, on a bad file. Kill a worker that
        is frozen (see STUCK_WORKER_SECONDS), report the video of any worker
        that died, and start a fresh worker in its place."""
        nonlocal next_worker_id
        now = time.monotonic()
        for worker_id, process in list(processes.items()):
            if process.is_alive():
                started = current_video.get(worker_id, (None, now))[1]
                if now - started > STUCK_WORKER_SECONDS and worker_id not in timed_out:
                    timed_out.add(worker_id)
                    process.kill()  # noticed as dead on a later pass
                continue
            del processes[worker_id]
            if process.exitcode == 0:
                continue  # finished normally; its "exit" message is still on the way
            video_id, _ = current_video.pop(worker_id, (None, None))
            if video_id is not None:
                result = videos.setdefault(video_id, VideoResult())
                if worker_id in timed_out:
                    result.error = f"decoder froze: no progress for {STUCK_WORKER_SECONDS} s"
                else:
                    result.error = f"decoder process crashed (exit code {process.exitcode})"
                result.crashed = True
            processes[next_worker_id] = start_worker(next_worker_id)
            next_worker_id += 1

    def close_crashed_videos():
        """Called when nothing is in flight: a crashed video will receive no
        more frames, so count what arrived and finish it."""
        if gpu.in_flight or not batcher.is_empty():
            return
        for video_id, result in list(videos.items()):
            if result.crashed and result.frames_expected is None:
                result.frames_expected = result.frames_counted
                result.n_seconds = max(result.frame_seconds, default=-1) + 1
                finish_if_complete(video_id)

    try:
        while processes or reader.is_alive() or not batcher.is_empty() or gpu.in_flight:
            t0 = time.perf_counter()
            try:
                message = results.get(timeout=0.01)
            except queue.Empty:
                message = None
            stats.wait_seconds += time.perf_counter() - t0

            if message is not None:
                handle(message)
                for items in batcher.full_batches():
                    gpu.submit(items)
            else:
                # Nothing arrived. If the GPU has nothing to do either, give
                # it the part-filled batches rather than let it sit idle.
                if not gpu.in_flight:
                    for items in batcher.all_batches():
                        gpu.submit(items)
                replace_crashed_workers()
                close_crashed_videos()

            for items, counts in gpu.finished():
                apply_counts(items, counts)
            stats.report(len(dispatched) if not reader.is_alive() else None)

        # Drain anything left (e.g. "done" messages queued after the last exit).
        while True:
            try:
                handle(results.get(timeout=0.5))
            except queue.Empty:
                break
        for items in batcher.all_batches():
            gpu.submit(items)
        for items, counts in gpu.finished(wait=True):
            apply_counts(items, counts)
        close_crashed_videos()

        # Safety net: any video handed out that never produced a line.
        for video_id in dispatched:
            if video_id not in written:
                result = videos.pop(video_id, VideoResult())
                result.error = result.error or "video was not processed"
                # keep any seconds that were counted before things went wrong
                result.n_seconds = max(result.frame_seconds, default=-1) + 1
                write_video(video_id, result)
    finally:
        gpu.close()
        out.close()
        for process in processes.values():
            process.terminate()
        # At exit, Python waits until everything put on a multiprocessing
        # queue has been delivered. If we got here because of an error, the
        # workers are gone and nobody will read, so that wait never ends and
        # the program hangs instead of exiting (this happened once, for
        # hours). Tell both queues not to wait.
        tasks.cancel_join_thread()
        results.cancel_join_thread()

    stats.report(len(dispatched), force=True)
    elapsed = time.perf_counter() - stats.start
    print(f"Done: {len(written)} videos in {elapsed:.1f}s -> {output_path}", file=sys.stderr)
