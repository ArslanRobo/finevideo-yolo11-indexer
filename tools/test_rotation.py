"""Check that rotated videos are turned upright exactly as FFmpeg does it.

    python tools/test_rotation.py        (needs the ffmpeg command)

Makes a small test video, tags copies of it with a rotation of 90, 180 and
270 degrees, and compares our decoded first frame with FFmpeg's
auto-rotated one. They must be pixel-identical. (A first version had the
direction of the 90/270 turns reversed; this test caught it.)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from videoindex.decode import frame_to_bgr, iter_frames  # noqa: E402


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tmp / "base.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x180:rate=10", "-t", "1",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(base))
        for degrees in (0, 90, 180, 270):
            tagged = tmp / f"rot{degrees}.mp4"
            ffmpeg("-display_rotation", str(degrees), "-i", str(base), "-c", "copy", str(tagged))
            raw = tmp / f"rot{degrees}.raw"
            ffmpeg("-i", str(tagged), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", str(raw))

            _, frame = next(iter_frames(tagged.read_bytes()))
            ours = frame_to_bgr(frame)
            theirs = np.fromfile(raw, np.uint8)
            assert theirs.size == ours.size, f"{degrees}: size differs"
            assert np.array_equal(ours, theirs.reshape(ours.shape)), f"{degrees}: pixels differ"
            print(f"ok   rotation {degrees:3d}: {ours.shape[1]}x{ours.shape[0]}, identical to ffmpeg")


if __name__ == "__main__":
    main()
