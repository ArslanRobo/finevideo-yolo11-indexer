"""Turn per-frame detections into the per-second output lines.

These are the rules from section 5 of the brief, applied to whichever frames
of a second we ran the model on:

* an object is present in a second if it was found in at least half of
  that second's frames;
* the people count is the median of the per-frame person counts, rounded
  down when it falls halfway between two numbers.

The reference uses every frame; run.py uses a sample of frames per second.
Both call the same functions, so the rules can only differ in which frames
go in, never in how they are combined.
"""

from collections import defaultdict

import numpy as np

from .config import OBJECT_NAMES, PERSON_COLUMN


def summarize_second(frame_counts):
    """Summarize one second.

    frame_counts: int array of shape (n_frames, 5), one row per frame and one
    column per object in OBJECT_NAMES, holding how many of that object the
    model found in that frame.

    Returns (objects, people): the list of present object names and the
    people count.
    """
    n_frames = len(frame_counts)
    if n_frames == 0:
        return [], 0

    # Present if found in at least half of the frames: hits / n >= 1/2.
    frames_with_object = (frame_counts > 0).sum(axis=0)
    objects = [
        name
        for name, hits in zip(OBJECT_NAMES, frames_with_object)
        if 2 * hits >= n_frames
    ]

    # Median, rounding a half down: for an even count, average the two middle
    # values and floor. Integer arithmetic avoids float surprises.
    people = np.sort(frame_counts[:, PERSON_COLUMN])
    middle = n_frames // 2
    if n_frames % 2 == 1:
        median = int(people[middle])
    else:
        median = (int(people[middle - 1]) + int(people[middle])) // 2

    return objects, median


def video_lines(video_id, frame_seconds, frame_counts, n_seconds):
    """Build every output line for one video.

    frame_seconds: the second each processed frame belongs to.
    frame_counts:  matching (n_frames, 5) count rows.
    n_seconds:     one more than the second holding the video's last frame,
                   so the lines run from second 0 to n_seconds - 1.

    A second with no processed frames gets an empty line, as the brief asks.
    """
    rows_by_second = defaultdict(list)
    for second, counts in zip(frame_seconds, frame_counts):
        rows_by_second[int(second)].append(counts)

    lines = []
    for second in range(n_seconds):
        rows = rows_by_second.get(second)
        if rows:
            objects, people = summarize_second(np.asarray(rows))
        else:
            objects, people = [], 0
        lines.append(
            {"video": video_id, "second": second, "objects": objects, "people": people}
        )
    return lines


def error_line(video_id, message):
    return {"video": video_id, "error": message}
