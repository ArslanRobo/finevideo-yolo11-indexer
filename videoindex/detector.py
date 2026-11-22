"""YOLO11l on batches of frames, returning per-frame object counts.

Ultralytics' own predict() does the same work, one Python call per batch with
CPU-side preprocessing and Results objects we do not need. Here we do just:

  1. letterbox on the CPU, in the decode workers (`make_letterbox`), using
     Ultralytics' own LetterBox class so the model input is identical to
     what predict() would build;
  2. the model forward pass: a TensorRT FP16 engine if prepare.py built
     one (see trt_engine.py), otherwise the PyTorch model in fp16;
  3. preprocessing (BGR->RGB, HWC->CHW, uint8 -> float, /255) happens on
     the GPU: inside the TensorRT engine, or in `preprocess` for PyTorch;
  4. NMS with the brief's settings: a batched re-implementation of
     Ultralytics' non_max_suppression (see batched_nms_counts);
  5. count detections per target class per frame.

Steps 3 and 4 mirror ultralytics/engine/predictor.py `preprocess` and
ultralytics/utils/nms.py `non_max_suppression`.
"""

import sys

import torch
import torchvision
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from .config import CLASS_IDS, CONFIDENCE, ENGINE, IMAGE_SIZE, NMS_IOU, WEIGHTS

# yolo11l.pt has a maximum stride of 32; predict() letterboxes to a multiple of it.
MODEL_STRIDE = 32


def make_letterbox():
    """The same resize-and-pad predict() applies to a batch of equal-size frames.

    auto=True pads only up to the next multiple of the stride (a 640x360
    frame becomes 640x384, not 640x640). That is what predict() does for a
    .pt model when every image in the batch has the same shape, which is
    always true for frames of one video.
    """
    return LetterBox(new_shape=(IMAGE_SIZE, IMAGE_SIZE), auto=True, stride=MODEL_STRIDE)


class Detector:
    """backend: "tensorrt", "pytorch", or "auto" (TensorRT if the engine
    file exists and loads, otherwise PyTorch)."""

    def __init__(self, device="cuda:0", half=True, backend="auto", weights=WEIGHTS, engine=ENGINE):
        self.device = torch.device(device)
        self.half = half
        self.backend = None

        if backend in ("auto", "tensorrt"):
            try:
                from .trt_engine import TrtModel

                self.model = TrtModel(engine, self.device)
                self.backend = "tensorrt"
            except Exception as exc:
                if backend == "tensorrt":
                    raise
                print(f"TensorRT engine not used ({type(exc).__name__}: {exc}); "
                      f"falling back to PyTorch", file=sys.stderr)

        if self.backend is None:
            model = YOLO(str(weights)).model  # the underlying torch nn.Module
            model = model.fuse(verbose=False)  # predict() also fuses Conv+BN
            model = model.to(self.device).eval()
            if half:
                model = model.half()
            self.model = model
            self.backend = "pytorch-fp16" if half else "pytorch-fp32"

        self.stream = torch.cuda.Stream(self.device)

        # Lookup: COCO class id -> column (0..4) in our count rows.
        self._column_of_class = torch.full((256,), -1, dtype=torch.long, device=self.device)
        for column, class_id in enumerate(CLASS_IDS):
            self._column_of_class[class_id] = column

    def count_objects(self, frames):
        """frames: uint8 tensor (B, H, W, 3), letterboxed, BGR, on any device.

        Returns an int64 CPU tensor (B, 5): for each frame, how many of each
        target object were detected with confidence >= CONFIDENCE.

        If the GPU runs out of memory (it happened on a shared development
        GPU), the batch is split in half and retried, down to one frame,
        instead of failing the whole run. Results are the same either way:
        each frame is processed independently.
        """
        try:
            # A stream of our own: TensorRT adds extra synchronisation when
            # it is given CUDA's default stream.
            with torch.cuda.stream(self.stream):
                return self._count_batch(frames)
        except torch.OutOfMemoryError:
            if len(frames) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(frames) // 2
            return torch.cat([self.count_objects(frames[:half]), self.count_objects(frames[half:])])

    @torch.inference_mode()
    def _count_batch(self, frames):
        predictions = self.raw_predictions(frames)
        return batched_nms_counts(predictions, self._column_of_class).cpu()

    @torch.inference_mode()
    def raw_predictions(self, frames):
        """Model output (B, 84, anchors) for uint8 BGR frames (B, H, W, 3)."""
        frames = frames.to(self.device, non_blocking=True)
        if self.backend == "tensorrt":
            return self.model(frames)  # the engine does its own preprocessing
        dtype = torch.float16 if self.backend == "pytorch-fp16" else torch.float32
        return self.model(preprocess(frames, dtype))


def preprocess(frames, dtype):
    """uint8 BGR (B, H, W, 3) -> float RGB (B, 3, H, W) in 0..1, for the
    PyTorch model. The TensorRT engine has the same steps built in."""
    x = frames.flip(-1)                # BGR -> RGB (the model was trained on RGB)
    x = x.permute(0, 3, 1, 2)          # BHWC -> BCHW
    return x.to(dtype).contiguous() / 255.0


MAX_DETECTIONS = 300  # per image, Ultralytics' max_det default


def batched_nms_counts(predictions, column_of_class):
    """NMS for a whole batch in one go, then per-frame counts of our 5 objects.

    Gives the same detections as Ultralytics'
    `non_max_suppression(predictions, conf_thres=0.4, iou_thres=0.7,
    classes=CLASS_IDS, max_det=300)` (ultralytics/utils/nms.py), which loops
    over the images in Python and took as long as the model itself
    (26.8 ms vs 26.1 ms for a batch of 64). This does the same steps for all
    images at once:

      1. each box takes its single best class over ALL 80 COCO classes
         (multi_label=False), and is kept if that score is > 0.4 (strictly
         greater, as in Ultralytics) and that class is one of our five;
      2. class-aware NMS at IoU 0.7: boxes only suppress boxes of the same
         class IN THE SAME IMAGE, so the NMS group is (image, class);
         torchvision's batched_nms is the same torchvision NMS Ultralytics
         calls, with the same coordinate-offset trick to keep groups apart;
      3. at most 300 detections per image, highest scores first.

    tools/test_nms.py checks the counts against Ultralytics on real frames.

    predictions: raw model output (B, 4 + 80, anchors), boxes as center x,
                 center y, width, height (or a tuple whose first item is that).
    Returns an int64 tensor (B, 5) on the predictions' device.
    """
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    n_images = predictions.shape[0]
    n_columns = int((column_of_class >= 0).sum())
    pred = predictions.transpose(1, 2)            # (B, anchors, 84)

    # 1. best class per box, then the confidence and class filters
    best_score, best_class = pred[..., 4:].max(dim=-1)
    keep = (best_score > CONFIDENCE) & (column_of_class[best_class] >= 0)
    image, anchor = keep.nonzero(as_tuple=True)
    if image.numel() == 0:
        return torch.zeros((n_images, n_columns), dtype=torch.long, device=pred.device)

    boxes = xywh2xyxy(pred[image, anchor, :4])
    scores = best_score[image, anchor]
    classes = best_class[image, anchor]

    # 2. one NMS call for every (image, class) group at once
    n_classes = pred.shape[-1] - 4
    groups = image * n_classes + classes
    kept = torchvision.ops.batched_nms(boxes.float(), scores.float(), groups, NMS_IOU)  # sorted by score, high first
    kept_image = image[kept]

    # 3. cap each image at MAX_DETECTIONS, keeping its highest scores: group
    #    the kept boxes by image (a stable sort keeps the score order inside
    #    each image) and number them 0, 1, 2, ... within their image.
    order = torch.argsort(kept_image, stable=True)
    sorted_image = kept_image[order]
    first_of_image = torch.searchsorted(sorted_image, sorted_image)
    rank = torch.arange(len(order), device=pred.device) - first_of_image
    kept = kept[order[rank < MAX_DETECTIONS]]

    # Count per (image, column).
    flat = image[kept] * n_columns + column_of_class[classes[kept]]
    counts = torch.bincount(flat, minlength=n_images * n_columns)
    return counts.view(n_images, n_columns)


def xywh2xyxy(xywh):
    """(center x, center y, width, height) -> (x1, y1, x2, y2), as in Ultralytics."""
    xy, wh = xywh[..., :2], xywh[..., 2:] / 2
    return torch.cat((xy - wh, xy + wh), dim=-1)
