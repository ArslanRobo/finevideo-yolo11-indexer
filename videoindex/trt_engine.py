"""YOLO11l as a TensorRT FP16 engine: building it, and running it.

Why: the GPU is the bottleneck (busy ~93% of the run), and TensorRT runs the
same network much faster than PyTorch by fusing layers and picking the
fastest kernels for this exact GPU.

How it is made (once, in prepare.py, on the machine that will run it,
because an engine only works on the GPU model and TensorRT version that
built it):

  yolo11l.pt --Ultralytics export--> yolo11l.onnx
             --add preprocessing-->  yolo11l_preprocessed.onnx
             --TensorRT build-->     yolo11l.engine

We build the engine ourselves rather than with Ultralytics' `format="engine"`
export because that one tunes the engine for 640x640 inputs, while 3 of 4
FineVideo frames letterbox to 640 wide x 384 high. Here the engine is tuned
for 384x640 and accepts any size up to 640x640 and any batch up to
MAX_BATCH.

The preprocessing is part of the engine: it takes the letterboxed frames
exactly as the decode workers produce them, uint8 (B, H, W, 3) in BGR
order, and does uint8 -> float, HWC -> CHW, BGR -> RGB and /255 itself
(`add_preprocessing`). It returns the same raw output as the PyTorch model,
(B, 84, anchors), which goes to the same NMS.
"""

import torch

from .config import IMAGE_SIZE

MAX_BATCH = 64
TUNED_SHAPE = (384, IMAGE_SIZE)          # (H, W) of the most common frames
MAX_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)     # letterboxing never exceeds 640 per side
MIN_SIDE = 32                            # the model's stride


def export_onnx(weights_path):
    """yolo11l.pt -> yolo11l.onnx with dynamic batch, height and width."""
    from ultralytics import YOLO

    onnx_path = YOLO(str(weights_path)).export(
        format="onnx", dynamic=True, simplify=True, imgsz=IMAGE_SIZE
    )
    return onnx_path


def add_preprocessing(onnx_path, out_path):
    """Put the input preprocessing inside the ONNX graph.

    New input "frames": uint8 (batch, height, width, 3), BGR. Four nodes turn
    it into what the exported model expects as "images" (float32
    (batch, 3, height, width), RGB, 0..1):

        Cast(float32) -> Transpose(to B,3,H,W) -> Gather([2,1,0]) -> Div(255)

    The Cast comes FIRST on purpose: TensorRT rejects a Gather (the channel
    flip) on uint8 data but accepts a Cast from uint8. The result is the
    same arithmetic as the PyTorch path in detector.preprocess().
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    old_input = graph.input[0]  # "images", float32 (batch, 3, height, width)
    model_input = old_input.name

    frames = helper.make_tensor_value_info("frames", TensorProto.UINT8, ["batch", "height", "width", 3])
    constants = [
        numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), "bgr_to_rgb"),
        numpy_helper.from_array(np.array(255.0, dtype=np.float32), "pixel_scale"),
    ]
    preprocessing = [
        helper.make_node("Cast", ["frames"], ["frames_float"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["frames_float"], ["frames_chw"], perm=[0, 3, 1, 2]),
        helper.make_node("Gather", ["frames_chw", "bgr_to_rgb"], ["frames_rgb"], axis=1),
        helper.make_node("Div", ["frames_rgb", "pixel_scale"], [model_input]),
    ]

    graph.input.remove(old_input)
    graph.input.insert(0, frames)
    graph.initializer.extend(constants)
    nodes = preprocessing + list(graph.node)  # new nodes first: the graph must stay in order
    graph.ClearField("node")
    graph.node.extend(nodes)

    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    return out_path


def build_engine(onnx_path, engine_path):
    """Build an FP16 TensorRT engine from the ONNX file. Takes a few minutes."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("could not parse ONNX: " + "; ".join(errors))

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # layers run in FP16; input uint8, output FP32
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # One optimization profile covering every input this pipeline can send.
    # Shapes are (batch, height, width, 3): the frames as the workers send them.
    profile = builder.create_optimization_profile()
    name = network.get_input(0).name
    profile.set_shape(
        name,
        min=(1, MIN_SIDE, MIN_SIDE, 3),
        opt=(MAX_BATCH, *TUNED_SHAPE, 3),
        max=(MAX_BATCH, *MAX_SHAPE, 3),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)


class TrtModel:
    """Runs a built engine on CUDA torch tensors, as a drop-in for the
    PyTorch model's forward pass."""

    def __init__(self, engine_path, device):
        import tensorrt as trt

        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not load TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            # TensorRT returns None rather than raising, e.g. when the GPU has
            # too little free memory for the context's working memory (~2.4 GB
            # for batch 64 at 640x640). Raise so the Detector falls back to
            # PyTorch instead of failing on the first batch.
            raise RuntimeError("could not create a TensorRT execution context (out of GPU memory?)")

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

        # An engine from before the preprocessing was baked in takes float
        # input; refuse it rather than feed it raw pixels.
        if self.engine.get_tensor_dtype(self.input_name) != trt.DataType.UINT8:
            raise RuntimeError("engine does not take uint8 frames; rebuild it with prepare.py")

    def __call__(self, frames):
        """frames: uint8 tensor (B, H, W, 3), letterboxed, BGR, on the GPU.
        B <= MAX_BATCH, H and W <= 640."""
        x = frames.contiguous()
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        out = torch.empty(
            tuple(self.context.get_tensor_shape(self.output_name)), dtype=torch.float32, device=self.device
        )
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        # Run on PyTorch's current CUDA stream, so the work is ordered with
        # the preprocessing before it and the NMS after it.
        if not self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return out
