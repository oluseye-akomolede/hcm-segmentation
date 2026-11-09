"""Reading, validating and running a TensorRT engine exported from a demo checkpoint.

This module owns everything the demos share about an engine: the ``engine.json`` record,
the refusals that keep a wrong engine from being served, and the forward itself. The
TensorRT import is inside the functions, so this module -- and every refusal in it -- can
be imported and tested on a host with no TensorRT at all, which is the CPU CI case.

No pycuda on the device: inputs and outputs are torch CUDA tensors, addressed by their
``data_ptr()``, and execution is queued on torch's current stream so the kernels land in
the order torch already established.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ENGINE_JSON = "engine.json"
ENGINE_FILE = "model.engine"
ONNX_FILE = "model.onnx"

# The keys a loader needs to be able to refuse a wrong engine. The input dtype and the
# output names/shapes/dtypes are included so a loader can say what to feed the engine and
# what comes back, without querying TensorRT first.
REQUIRED_KEYS = (
    "source_sha256",
    "input_size",
    "precision",
    "tensorrt_version",
    "input_dtype",
    "outputs",
)


class EngineError(Exception):
    """A refusal from the engine loader or the exporter: one line, exit 2."""


class EngineUnavailable(EngineError):
    """TensorRT (or the ONNX side of it) is not importable on this host."""


def _import_trt():
    try:
        import tensorrt as trt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise EngineUnavailable(
            "the TensorRT Python bindings are not importable on this host; "
            "on the Jetson they live in /usr/lib/python3.10/dist-packages "
            f"({exc})"
        ) from exc
    return trt


def sha256_file(path, *, chunk_size: int = 1 << 20) -> str:
    """The sha256 of a file, read in chunks so a large checkpoint is not held in RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_engine_json(directory) -> dict:
    """Parse ``engine.json`` under ``directory``, refusing anything unusable.

    A missing record, an unreadable one, one that is not JSON, one that is not an object,
    and one missing a required key are all refusals: a loader that fell back to the engine
    file would serve an engine whose provenance it cannot check, which is the failure this
    record exists to prevent.
    """
    path = Path(directory) / ENGINE_JSON
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EngineError(
            f"no {ENGINE_JSON} in {directory}: this directory is not an exported engine "
            f"(run jetson/export_trt.py to make one)"
        ) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise EngineError(f"cannot read {path}: {exc}") from exc
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EngineError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise EngineError(
            f"{path} must be a JSON object, got {type(record).__name__}"
        )
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        raise EngineError(f"{path} is missing {', '.join(missing)}")
    return record


def validate_engine(record: dict, *, input_size=None, source_sha256=None, where=ENGINE_JSON) -> dict:
    """Refuse a record that does not match what the caller is about to serve.

    ``input_size`` and ``source_sha256`` are what the caller states it will serve; ``None``
    means the caller cannot check that field, not that anything matches.
    """
    recorded_size = record.get("input_size")
    if input_size is not None and int(recorded_size) != int(input_size):
        raise EngineError(
            f"{where} was built for input size {recorded_size}, but this run serves {input_size}"
        )
    recorded_sha = record.get("source_sha256")
    if source_sha256 is not None and str(recorded_sha) != str(source_sha256):
        raise EngineError(
            f"{where} was built from checkpoint {recorded_sha}, but this run serves {source_sha256}"
        )
    return record


def load_engine_record(directory, *, input_size=None, source_sha256=None) -> dict:
    """Read and validate the record under ``directory``."""
    record = read_engine_json(directory)
    return validate_engine(
        record,
        input_size=input_size,
        source_sha256=source_sha256,
        where=f"{directory}/{ENGINE_JSON}",
    )


def _torch_dtype(trt, trt_dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if trt_dtype not in mapping:
        raise EngineError(f"unsupported TensorRT tensor dtype {trt_dtype}")
    return mapping[trt_dtype]


def dtype_name(dtype: torch.dtype) -> str:
    """``"float16"`` for ``torch.float16``: the name ``engine.json`` records."""
    return str(dtype).rsplit(".", 1)[-1]


class TrtEngine:
    """A loaded engine, run through torch CUDA buffers on torch's current stream."""

    def __init__(self, record: dict, *, runtime, engine, context):
        self.record = record
        self._runtime = runtime
        self._engine = engine
        self._context = context
        self.input_name = record.get("input_name", "input")
        self.output_names = [out["name"] for out in record["outputs"]]

    @property
    def input_size(self) -> int:
        return int(self.record["input_size"])

    @property
    def precision(self) -> str:
        return str(self.record.get("precision", "unknown"))

    @classmethod
    def load(cls, directory, *, input_size=None, source_sha256=None) -> "TrtEngine":
        """Load ``directory``'s engine, refusing one that does not match the stated serve."""
        record = load_engine_record(directory, input_size=input_size, source_sha256=source_sha256)
        engine_path = Path(directory) / ENGINE_FILE
        try:
            raw = engine_path.read_bytes()
        except OSError as exc:
            raise EngineError(f"cannot read {engine_path}: {exc}") from exc
        trt = _import_trt()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(raw)
        if engine is None:
            raise EngineError(
                f"TensorRT could not deserialize {engine_path}: it is corrupt or was built "
                f"by an incompatible TensorRT"
            )
        context = engine.create_execution_context()
        return cls(record, runtime=runtime, engine=engine, context=context)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``x`` (a CUDA [1,3,H,W] tensor) and return the first output tensor.

        The engine is shape-specialised to this checkpoint's input size; a caller that
        passes another shape gets a TensorRT error rather than a silently resized result.

        The engine also declares the dtype of its input tensor (an fp16 build declares
        fp16), and TensorRT reads that buffer at the declared width -- so a float32 tensor
        handed to an fp16 engine would be reinterpreted, not refused. ``x`` is converted to
        the engine's dtype first.
        """
        trt = _import_trt()
        input_dtype = _torch_dtype(trt, self._engine.get_tensor_dtype(self.input_name))
        if x.dtype != input_dtype:
            x = x.to(dtype=input_dtype)
        x = x.contiguous()
        self._context.set_input_shape(self.input_name, tuple(x.shape))
        outputs = []
        for out in self.record["outputs"]:
            name = out["name"]
            shape = tuple(self._context.get_tensor_shape(name))
            dtype = _torch_dtype(trt, self._engine.get_tensor_dtype(name))
            outputs.append((name, torch.empty(shape, dtype=dtype, device=x.device)))
        self._context.set_tensor_address(self.input_name, x.data_ptr())
        for name, tensor in outputs:
            self._context.set_tensor_address(name, tensor.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        if not self._context.execute_async_v3(stream):
            raise EngineError("TensorRT execution failed (execute_async_v3 returned false)")
        return outputs[0][1]
