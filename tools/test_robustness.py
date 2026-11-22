"""Check that a stuck decoder or a failing reader cannot hang the run.

    python tools/test_robustness.py <folder of .parquet files (small)>

Three scenarios, each run in a fresh process with a 5-minute limit. In
each the run must FINISH (not hang), and in 1 and 2 every video must still
have a line.

1. Slow video: VIDEO_TIMEOUT_SECONDS = 0.05 s, so workers give up on videos
   partway (finished seconds kept, plus an error line).
2. Frozen decoder: decoding row 1 blocks forever inside the decoder, where
   the worker cannot check the clock; with STUCK_WORKER_SECONDS = 5 the main
   process must kill that worker, report the video, and carry on.
3. Reader failure: the Parquet reader raises after the first video; the
   workers must still get their stop signals. (Videos the reader never got
   to cannot have lines; the point is that the run ends.)

The patches sit at the top of the generated script, so they apply in the
spawned worker processes too (spawn re-imports the main script).
"""

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENARIOS = {
    "slow video": """
import videoindex.pipeline as p
p.VIDEO_TIMEOUT_SECONDS = 0.05
""",
    "frozen decoder": """
import time
import videoindex.pipeline as p
p.STUCK_WORKER_SECONDS = 5
real_iter_frames = p.iter_frames
def iter_frames(mp4_bytes, *args):
    if len(mp4_bytes) == FROZEN_SIZE:
        time.sleep(10**6)  # stuck inside the "decoder": cannot check the clock
    yield from real_iter_frames(mp4_bytes, *args)
p.iter_frames = iter_frames
""",
    "reader failure": """
import videoindex.pipeline as p
real = p.iter_videos
def failing(path):
    for i, item in enumerate(real(path)):
        if i == 1:
            raise RuntimeError("simulated reader failure")
        yield item
p.iter_videos = failing
""",
}

RUN = """
import sys
sys.path.insert(0, {root!r})
FROZEN_SIZE = {frozen_size}
{patch}
if __name__ == "__main__":
    p.run(input_dir={input!r}, output_path={output!r}, workers=2, batch_size=64,
          sample_rate=3, device="cuda:0", half=True)
"""


def main():
    import pyarrow.parquet as pq

    folder = sys.argv[1]
    first_file = sorted(Path(folder).glob("*.parquet"))[0]
    frozen_size = len(pq.read_table(first_file, columns=["mp4"]).column("mp4")[1].as_py())  # row 1
    for name, patch in SCENARIOS.items():
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            script = Path(tmp) / "scenario.py"  # a file, so spawned workers can import it
            script.write_text(RUN.format(root=str(ROOT), patch=patch, input=folder, output=str(output),
                                         frozen_size=frozen_size))
            try:
                proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"FAIL {name}: the run hung")
            assert proc.returncode == 0, f"{name}: exit {proc.returncode}\n{proc.stderr[-2000:]}"

            lines = defaultdict(list)
            for line in output.read_text().splitlines():
                row = json.loads(line)
                lines[row["video"]].append(row)
            errors = {v: r[-1]["error"] for v, r in lines.items() if "error" in r[-1]}
            print(f"ok   {name}: finished, {len(lines)} videos have lines, {len(errors)} with an error line")
            for video, message in sorted(errors.items())[:3]:
                print(f"       {video}: {message}")


if __name__ == "__main__":
    main()
