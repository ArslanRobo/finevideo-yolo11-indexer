"""Checks for the per-second rules and the frame sampler.

    python tools/test_rules.py

These pieces are shared by run.py and the reference, so a mistake in them
would not show up when comparing the two. They are checked here instead,
against hand-worked examples from the brief's wording.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from videoindex.decode import FrameSampler, second_of  # noqa: E402
from videoindex.seconds import summarize_second, video_lines  # noqa: E402


def counts(*people_and_objects):
    """Rows of (person, car, dog, laptop, cell phone) counts."""
    return np.array(people_and_objects)


def test_present_if_in_at_least_half():
    # car in 2 of 4 frames: exactly half -> present. dog in 1 of 4 -> absent.
    objects, _ = summarize_second(counts([0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]))
    assert objects == ["car"], objects


def test_person_object_follows_same_rule():
    objects, people = summarize_second(counts([1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]))
    assert objects == [] and people == 0, (objects, people)


def test_median_odd():
    _, people = summarize_second(counts([3, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0]))
    assert people == 2


def test_median_even_rounds_half_down():
    # middle values 1 and 2 -> 1.5 -> rounded down to 1
    _, people = summarize_second(counts([1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [0, 0, 0, 0, 0], [5, 0, 0, 0, 0]))
    assert people == 1, people
    # middle values 2 and 4 -> 3 exactly
    _, people = summarize_second(counts([2, 0, 0, 0, 0], [4, 0, 0, 0, 0]))
    assert people == 3, people


def test_object_order_matches_brief():
    objects, _ = summarize_second(counts([1, 1, 1, 1, 1]))
    assert objects == ["person", "car", "dog", "laptop", "cell phone"], objects


def test_every_second_gets_a_line():
    # frames only in seconds 0 and 3; seconds 1 and 2 must still appear, empty
    lines = video_lines("f.parquet:0", [0, 3], [[2, 0, 0, 0, 0], [0, 1, 0, 0, 0]], n_seconds=4)
    assert [line["second"] for line in lines] == [0, 1, 2, 3]
    assert lines[1] == {"video": "f.parquet:0", "second": 1, "objects": [], "people": 0}
    assert lines[3]["objects"] == ["car"]


def test_second_boundaries():
    # second 12 holds 12.0 <= t < 13.0
    assert second_of(12.0) == 12
    assert second_of(12.999) == 12
    assert second_of(13.0) == 13


def test_sampler_spreads_frames_evenly():
    # 30 fps, 5 samples a second -> keep every 6th frame
    sampler = FrameSampler(5)
    kept = [i for i in range(60) if sampler.keep(i / 30)]
    assert kept == list(range(0, 60, 6)), kept


def test_sampler_keeps_a_frame_in_every_second_even_at_low_fps():
    sampler = FrameSampler(5)
    # 2 fps video: every frame is the first of its slot
    kept = [i for i in range(10) if sampler.keep(i / 2)]
    assert kept == list(range(10))


def test_sampler_none_keeps_everything():
    sampler = FrameSampler(None)
    assert all(sampler.keep(i / 30) for i in range(100))


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} passed")
