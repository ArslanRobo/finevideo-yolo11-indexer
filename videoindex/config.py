"""Fixed settings from the brief, in one place.

Anything here is dictated by the assignment (sections 1, 3 and 5), not tuned.
Tunable knobs (workers, batch size, sampling rate) are command-line options
of run.py instead.
"""

from pathlib import Path

# Project root: the folder that holds run.py and, after prepare.py, the weights.
ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = ROOT / "yolo11l.pt"
ENGINE = ROOT / "yolo11l.engine"   # built by prepare.py; optional

# Section 3: input size 640, keep detections with confidence >= 0.4,
# class-aware NMS at IoU 0.7 (the Ultralytics default).
IMAGE_SIZE = 640
CONFIDENCE = 0.4
NMS_IOU = 0.7

# Section 1: the five objects, in the order they are written in the brief.
# Values are the COCO class ids used by yolo11l.pt.
TARGET_CLASSES = {
    "person": 0,
    "car": 2,
    "dog": 16,
    "laptop": 63,
    "cell phone": 67,
}
OBJECT_NAMES = list(TARGET_CLASSES)           # column order used everywhere
CLASS_IDS = list(TARGET_CLASSES.values())
PERSON_COLUMN = OBJECT_NAMES.index("person")  # == 0
