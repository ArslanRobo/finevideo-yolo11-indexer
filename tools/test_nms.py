"""Check the batched NMS against Ultralytics' own non_max_suppression.

    python tools/test_nms.py <a .parquet file>

Runs the model on real frames from the given file and compares, frame by
frame, the object counts from videoindex.detector.batched_nms_counts with
counts from Ultralytics' non_max_suppression on the SAME model output. Also
checks a synthetic crowded image with more than 300 boxes, for the
per-image cap.
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

from videoindex.config import CLASS_IDS, CONFIDENCE, NMS_IOU  # noqa: E402
from videoindex.decode import frame_to_bgr, iter_frames  # noqa: E402
from videoindex.detector import Detector, batched_nms_counts, make_letterbox  # noqa: E402


def ultralytics_counts(predictions, column_of_class):
    detections = non_max_suppression(predictions, conf_thres=CONFIDENCE, iou_thres=NMS_IOU,
                                     classes=CLASS_IDS, max_det=300)
    counts = torch.zeros((len(detections), len(CLASS_IDS)), dtype=torch.long)
    for i, det in enumerate(detections):
        for class_id in det[:, 5].long().tolist():
            counts[i, int(column_of_class[class_id])] += 1
    return counts


def batches_of_real_frames(parquet_path, n_videos=8, per_video=64):
    letterbox = make_letterbox()
    column = pq.read_table(parquet_path, columns=["mp4"]).column("mp4")
    for row in range(min(n_videos, len(column))):
        images = []
        for i, (_, frame) in enumerate(iter_frames(column[row].as_py())):
            if i % 7 == 0:
                images.append(letterbox(image=frame_to_bgr(frame)))
            if len(images) == per_video:
                break
        if images:
            yield row, torch.from_numpy(np.stack(images))


@torch.inference_mode()
def main():
    parquet_path = sys.argv[1]
    for backend in ("tensorrt", "pytorch"):
        detector = Detector(backend=backend)
        frames_checked = detections = mismatched = 0
        for row, frames in batches_of_real_frames(parquet_path):
            predictions = detector.raw_predictions(frames)
            ours = batched_nms_counts(predictions, detector._column_of_class).cpu()
            theirs = ultralytics_counts(predictions, detector._column_of_class.cpu())
            mismatched += int((ours != theirs).any(dim=1).sum())
            frames_checked += len(frames)
            detections += int(theirs.sum())
        print(f"{detector.backend}: {frames_checked} real frames, {detections} detections, "
              f"{mismatched} frame(s) with different counts")
        assert mismatched == 0

    # Crowded synthetic image: 400 separate person boxes, all confident.
    # Ultralytics keeps the 300 best; so must we.
    column_of_class = detector._column_of_class
    pred = torch.zeros((1, 84, 400), device="cuda")
    pred[0, 0] = torch.arange(400, device="cuda") * 20.0 + 10   # centers far apart: no overlap
    pred[0, 1] = 100
    pred[0, 2] = pred[0, 3] = 8                                  # 8x8 boxes
    pred[0, 4 + 0] = torch.linspace(0.5, 0.99, 400, device="cuda")  # class 0 = person
    ours = batched_nms_counts(pred, column_of_class).cpu()
    theirs = ultralytics_counts(pred, column_of_class.cpu())
    print(f"crowded image: ours {ours.tolist()}, ultralytics {theirs.tolist()}")
    assert torch.equal(ours, theirs) and ours[0, 0] == 300
    print("ok")


if __name__ == "__main__":
    main()
