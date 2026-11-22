"""Decode an mp4 held in memory and pick the frames to run the model on.

Timestamps follow section 4 of the brief: seconds since the FIRST DECODED
frame, from each frame's presentation timestamp (pts). Second N holds the
frames with N <= t < N + 1.

Every frame is decoded (H.264 cannot skip frames others depend on), but only
the sampled ones are converted to a numpy image, which is the costly part.
"""

import io
import math

import av
import numpy as np


def iter_frames(mp4_bytes, decoder_threads=1):
    """Yield (t, frame) for every frame of the first video stream, in
    presentation order. t is seconds since the first decoded frame.
    """
    with av.open(io.BytesIO(mp4_bytes), mode="r") as container:
        if not container.streams.video:
            raise ValueError("no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # frame + slice threading inside the decoder
        stream.codec_context.thread_count = decoder_threads

        time_base = stream.time_base
        # Used only when a frame has no pts: step forward by one frame period.
        frame_period = 1.0 / float(stream.average_rate or stream.guessed_rate or 30)

        first_pts = None
        t = 0.0
        for frame in container.decode(stream):
            if frame.pts is None or time_base is None:
                t = t + frame_period if first_pts is not None else 0.0
                if first_pts is None:
                    first_pts = 0
            else:
                if first_pts is None:
                    first_pts = frame.pts
                # Integer pts difference times the time base. For a frame exactly
                # on a second boundary this is exact (e.g. 360 * 1/30 == 12.0).
                t = (frame.pts - first_pts) * time_base.numerator / time_base.denominator
            # A frame presented before the first decoded one (rare, broken
            # streams) still belongs to the video; count it in second 0.
            yield max(t, 0.0), frame


def frame_to_bgr(frame):
    """Convert a decoded frame to a BGR uint8 image, upright.

    If the video carries a rotation tag (a display matrix), the frame is
    rotated the way a player would show it, as section 3 asks.
    """
    image = frame.to_ndarray(format="bgr24")
    quarter_turns_ccw = _quarter_turns_to_display(frame)
    if quarter_turns_ccw:
        image = np.ascontiguousarray(np.rot90(image, k=quarter_turns_ccw))
    return image


def _quarter_turns_to_display(frame):
    # PyAV's frame.rotation is the counterclockwise turn needed to show the
    # frame upright; np.rot90(k) turns counterclockwise by k quarter turns.
    # Checked against FFmpeg's own autorotate for tags of 90, 180 and 270
    # degrees (identical pixels) - the sign is easy to get wrong.
    angle = getattr(frame, "rotation", 0) or 0
    return int(round(angle / 90.0)) % 4


class FrameSampler:
    """Choose which frames to run the model on.

    Each second is split into `per_second` equal slots and the first frame
    that falls into each slot is kept. per_second=None keeps every frame.

    The result per second is then a small, evenly spread sample of that
    second's frames, and every second that has any frame gets at least one
    sampled frame, so no second is left without evidence.
    """

    def __init__(self, per_second=None):
        self.per_second = per_second
        self._last_slot = -1

    def keep(self, t):
        if self.per_second is None:
            return True
        slot = math.floor(t * self.per_second)
        if slot > self._last_slot:
            self._last_slot = slot
            return True
        return False


def second_of(t):
    return math.floor(t)
