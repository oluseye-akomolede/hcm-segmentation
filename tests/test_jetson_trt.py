"""The TensorRT engine path: the record, the refusals, and the wiring.

None of this needs TensorRT or a CUDA device: the loader's validation is pure
Python and runs before the engine is touched, which is the point. The one test
that does need TensorRT skips without it. ``jetson.realtime_demo`` is not
imported (it needs cv2, which this VM has no copy of), so its wiring is read as
text, the way ``tests/test_demo_input_size.py`` reads it.
"""
import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest
import torch

from jetson import bench_model, export_trt, trt_engine
from jetson.trt_engine import (
    ENGINE_FILE,
    ENGINE_JSON,
    EngineError,
    EngineUnavailable,
    TrtEngine,
    read_engine_json,
    sha256_file,
    validate_engine,
)

REALTIME_SRC = (Path(__file__).resolve().parents[1] / "jetson" / "realtime_demo.py").read_text()


def _has_tensorrt():
    try:
        import tensorrt  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        return False
    return True


def _record(**over):
    record = {
        "source_checkpoint": "/models/model.pth",
        "source_sha256": "a" * 64,
        "input_size": 352,
        "input_name": "input",
        "input_dtype": "float32",
        "precision": "fp32",
        "tensorrt_version": "10.3.0",
        "outputs": [{"name": "logits", "shape": [1, 2, 22, 22], "dtype": "float32"}],
    }
    record.update(over)
    return record


def _engine_dir(tmp_path, **over):
    """A directory that looks like an export, without any engine bytes being real."""
    directory = tmp_path / "engine"
    directory.mkdir(exist_ok=True)
    (directory / ENGINE_FILE).write_bytes(b"serialized-engine")
    (directory / ENGINE_JSON).write_text(json.dumps(_record(**over)))
    return directory


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------
def test_a_good_record_reads_back(tmp_path):
    directory = _engine_dir(tmp_path)
    assert read_engine_json(directory)["input_size"] == 352


def test_a_missing_record_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(EngineError, match=ENGINE_JSON):
        read_engine_json(tmp_path / "empty")


def test_an_unreadable_record_is_refused(tmp_path):
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / ENGINE_JSON).mkdir()  # reading a directory as text is an OSError
    with pytest.raises(EngineError, match="cannot read"):
        read_engine_json(directory)


def test_a_record_that_is_not_json_is_refused(tmp_path):
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / ENGINE_JSON).write_text("{not json,")
    with pytest.raises(EngineError, match="not valid JSON"):
        read_engine_json(directory)


def test_a_record_that_is_not_an_object_is_refused(tmp_path):
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / ENGINE_JSON).write_text("[1, 2, 3]")
    with pytest.raises(EngineError, match="must be a JSON object"):
        read_engine_json(directory)


# Spelled out rather than read off the tuple: a test parametrized over REQUIRED_KEYS stops
# testing a key the moment that key is dropped from the tuple, which is the regression.
REQUIRED_KEY_NAMES = (
    "source_sha256",
    "input_size",
    "precision",
    "tensorrt_version",
    "input_dtype",
    "outputs",
)


def test_the_loader_requires_exactly_the_keys_it_needs():
    assert set(trt_engine.REQUIRED_KEYS) == set(REQUIRED_KEY_NAMES)


def test_the_record_shape_does_not_have_to_carry_tf32():
    # `tf32` describes how the engine was built, it is not a key a loader needs to refuse a
    # wrong engine -- and an engine exported before the field existed has no `tf32` in its
    # record. Requiring it would refuse every engine on the box today, so `_record()` (which
    # has no `tf32`) has to keep reading back.
    assert "tf32" not in trt_engine.REQUIRED_KEYS
    assert "tf32" not in _record()


@pytest.mark.parametrize("key", REQUIRED_KEY_NAMES)
def test_a_record_missing_a_required_key_is_refused(tmp_path, key):
    # Dropping a key from REQUIRED_KEYS leaves a record that reads back here and raises a
    # bare KeyError in infer instead of being refused.
    directory = _engine_dir(tmp_path)
    record = _record()
    del record[key]
    (directory / ENGINE_JSON).write_text(json.dumps(record))
    with pytest.raises(EngineError, match=key):
        read_engine_json(directory)


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------
def test_the_input_size_must_match():
    with pytest.raises(EngineError) as raised:
        validate_engine(_record(input_size=192), input_size=352)
    assert "192" in str(raised.value) and "352" in str(raised.value)


def test_the_source_checkpoint_must_match():
    with pytest.raises(EngineError) as raised:
        validate_engine(_record(source_sha256="a" * 64), source_sha256="b" * 64)
    assert "a" * 8 in str(raised.value) and "b" * 8 in str(raised.value)


def test_an_engine_is_not_matched_against_a_field_the_caller_cannot_state():
    # None means "cannot check", not "anything matches", so nothing is refused.
    assert validate_engine(_record(), input_size=352)["input_size"] == 352


def test_a_loader_refuses_a_wrong_size_before_touching_tensorrt(tmp_path):
    directory = _engine_dir(tmp_path, input_size=192)
    with pytest.raises(EngineError) as raised:
        TrtEngine.load(directory, input_size=352)
    assert "192" in str(raised.value) and "352" in str(raised.value)


def test_a_loader_refuses_a_wrong_checkpoint_before_touching_tensorrt(tmp_path):
    directory = _engine_dir(tmp_path, source_sha256="a" * 64)
    with pytest.raises(EngineError, match="checkpoint"):
        TrtEngine.load(directory, source_sha256="b" * 64)


def test_a_bare_engine_file_is_not_run(tmp_path):
    directory = tmp_path / "bare"
    directory.mkdir()
    (directory / ENGINE_FILE).write_bytes(b"serialized-engine")
    with pytest.raises(EngineError, match=ENGINE_JSON):
        TrtEngine.load(directory)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob"
    payload = b"some weights" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# The engine forward matches the dtype the engine declares
# --------------------------------------------------------------------------
class _FakeContext:
    """Enough of an IExecutionContext for ``infer``: shapes, addresses, a stream."""

    def __init__(self, shapes):
        self.shapes = shapes
        self.addresses = {}
        self.input_shape = None
        self.stream = None

    def set_input_shape(self, name, shape):
        self.input_shape = tuple(shape)

    def get_tensor_shape(self, name):
        return self.shapes[name]

    def set_tensor_address(self, name, ptr):
        self.addresses[name] = ptr

    def execute_async_v3(self, stream):
        self.stream = stream
        return True


class _FakeEngine:
    def __init__(self, dtypes):
        self.dtypes = dtypes

    def get_tensor_dtype(self, name):
        return self.dtypes[name]


# Every dtype _torch_dtype names, as distinguishable sentinels.
_FAKE_TRT = types.SimpleNamespace(float32="trt-f32", float16="trt-f16", int32="trt-i32",
                                  int8="trt-i8", bool="trt-bool")


def _fake_infer(monkeypatch, *, input_dtype="trt-f32"):
    """A TrtEngine over a fake TRT/engine/context: no CUDA, no TensorRT, no engine bytes."""
    monkeypatch.setattr(trt_engine, "_import_trt", lambda: _FAKE_TRT)
    # infer queues on torch's current stream; a CPU-only host has no CUDA stream to ask for.
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: types.SimpleNamespace(cuda_stream=7))
    engine = _FakeEngine({"input": input_dtype, "logits": "trt-f32"})
    context = _FakeContext({"logits": [1, 2, 22, 22]})
    return TrtEngine(_record(), runtime=None, engine=engine, context=context), context


def test_the_engine_is_fed_the_dtype_it_declares(monkeypatch):
    # The fp16-engine case: a float32 pointer handed to an fp16 input tensor is read at the
    # wrong width, so infer must convert rather than pass the caller's buffer through.
    engine, context = _fake_infer(monkeypatch, input_dtype="trt-f16")
    converted = []
    real_to = torch.Tensor.to

    def spy(self, *args, **kwargs):
        out = real_to(self, *args, **kwargs)
        converted.append(out)
        return out

    monkeypatch.setattr(torch.Tensor, "to", spy)
    x = torch.zeros(1, 3, 22, 22, dtype=torch.float32)
    out = engine.infer(x)

    assert [t.dtype for t in converted] == [torch.float16]
    assert context.addresses["input"] == converted[0].data_ptr()  # TRT reads the cast buffer
    assert context.addresses["input"] != x.data_ptr()
    assert out.dtype == torch.float32  # the output dtype is still the engine's own


def test_an_input_already_in_the_engines_dtype_is_not_copied(monkeypatch):
    engine, context = _fake_infer(monkeypatch, input_dtype="trt-f32")
    x = torch.zeros(1, 3, 22, 22, dtype=torch.float32)
    engine.infer(x)
    assert context.addresses["input"] == x.data_ptr()


def test_the_recorded_dtype_names_are_torch_names():
    assert trt_engine.dtype_name(torch.float32) == "float32"
    assert trt_engine.dtype_name(torch.float16) == "float16"


# --------------------------------------------------------------------------
# The export CLI
# --------------------------------------------------------------------------
def _checkpoint(tmp_path, name="ckpt"):
    """A checkpoint directory with the weights file the export looks for."""
    ckpt = tmp_path / name
    ckpt.mkdir()
    (ckpt / "model.pth").write_bytes(b"weights")
    return ckpt


def _deps_present(monkeypatch):
    """Get a test past the dependency preflight: it is about the refusal behind it.

    This VM has neither ``onnx`` nor TensorRT, so without this every export refusal would
    be the dependency one. The preflight's own order is pinned by its own tests below.
    """
    monkeypatch.setattr(export_trt, "_require_export_deps", lambda: None)


def test_the_export_defaults_are_fp32_at_the_demo_geometry(tmp_path):
    args = export_trt._parse_args([str(tmp_path), "--out", str(tmp_path / "o")])
    assert args.fp16 is False
    assert args.opset == 17
    assert args.workspace_mb == 1024
    assert args.verify is False


def test_the_export_refuses_to_overwrite_an_existing_out(tmp_path, monkeypatch, capsys):
    _deps_present(monkeypatch)
    out = tmp_path / "already"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    rc = export_trt.main([str(_checkpoint(tmp_path)), "--out", str(out)])
    assert rc == 2
    assert "already exists" in capsys.readouterr().err
    assert (out / "keep.txt").read_text() == "mine"


def test_the_export_refuses_a_checkpoint_with_no_weights(tmp_path, monkeypatch, capsys):
    _deps_present(monkeypatch)
    empty = tmp_path / "ckpt"
    empty.mkdir()
    rc = export_trt.main([str(empty), "--out", str(tmp_path / "out")])
    assert rc == 2
    assert "no model.pth" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1"])
def test_the_export_refuses_a_workspace_below_one_mb(tmp_path, monkeypatch, capsys, value):
    _deps_present(monkeypatch)
    out = tmp_path / "out"
    rc = export_trt.main(
        [str(_checkpoint(tmp_path)), "--out", str(out), f"--workspace-mb={value}"])
    assert rc == 2
    assert "workspace-mb" in capsys.readouterr().err
    assert not out.exists()


def test_the_export_refuses_when_onnx_is_absent_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "onnx", None)  # `import onnx` now raises ImportError
    out = tmp_path / "out"
    rc = export_trt.main([str(_checkpoint(tmp_path)), "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err and "ONNX" in err
    assert not out.exists()


def test_the_export_refuses_when_tensorrt_is_absent_and_writes_nothing(tmp_path, monkeypatch, capsys):
    if _has_tensorrt():
        pytest.skip("TensorRT is present on this host")
    # onnx importable, TensorRT not: the ONNX half of the preflight passes, the refusal is
    # the TensorRT one, and it must land before the output directory exists. export_onnx is
    # deliberately not stubbed -- it is the call that used to run (and write) first.
    monkeypatch.setitem(sys.modules, "onnx", types.ModuleType("onnx"))
    out = tmp_path / "out"
    rc = export_trt.main([str(_checkpoint(tmp_path)), "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err and "TensorRT" in err
    assert not out.exists()


def test_the_dependency_check_comes_before_the_output_directory_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "onnx", None)
    out = tmp_path / "already"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    rc = export_trt.main([str(_checkpoint(tmp_path)), "--out", str(out)])
    assert rc == 2
    assert "ONNX" in capsys.readouterr().err  # the missing dependency, not "already exists"
    assert (out / "keep.txt").read_text() == "mine"


def test_building_without_tensorrt_is_a_refusal(tmp_path):
    if _has_tensorrt():
        pytest.skip("TensorRT is present on this host")
    with pytest.raises(EngineUnavailable):
        export_trt.build_engine(tmp_path / "model.onnx", fp16=False, workspace_mb=64)


def test_a_successful_export_writes_a_record_the_loader_accepts(tmp_path, monkeypatch):
    """The export's success path is untested: every other export test is a refusal.

    Drive ``export()`` to completion with the three expensive halves stubbed and read the
    record back through the loader's own reader, so a key renamed in the record (or the
    write dropped) fails here instead of on the box.
    """
    _deps_present(monkeypatch)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    weights = checkpoint / "model.pth"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(
        export_trt, "_load_checkpoint",
        lambda path: (types.SimpleNamespace(decoder_arch="v1"), weights, 352))
    monkeypatch.setattr(
        export_trt, "export_onnx",
        lambda model, input_size, onnx_path, opset: onnx_path.write_bytes(b"onnx-graph"))
    monkeypatch.setattr(
        export_trt, "build_engine",
        lambda onnx_path, fp16, workspace_mb: (b"serialized-engine", "10.3.0"))
    monkeypatch.setattr(
        export_trt, "engine_tensors",
        lambda engine_bytes, input_name, input_size: (
            "float32", [{"name": "logits", "shape": [1, 2, 22, 22], "dtype": "float32"}]))

    out = tmp_path / "out"
    assert export_trt.main([str(checkpoint), "--out", str(out)]) == 0

    assert (out / ENGINE_JSON).is_file()
    record = read_engine_json(out)  # enforces trt_engine.REQUIRED_KEYS on the written record
    assert record["input_size"] == 352
    assert record["precision"] == "fp32"
    # The record says how the engine's arithmetic was built, not just its nominal precision:
    # an fp32 engine built with TF32 tactics allowed is not fp32.
    assert record["tf32"] is False
    # The dtypes the record names are the engine's own. fp16 is refused for this graph, so
    # the exportable (and here, exported) input tensor is float32, not fp16.
    assert record["input_dtype"] == "float32"
    assert record["outputs"] == [
        {"name": "logits", "shape": [1, 2, 22, 22], "dtype": "float32"}]
    assert record["source_sha256"] == sha256_file(weights)
    assert record["onnx_sha256"] == sha256_file(out / export_trt.ONNX_FILE)
    assert record["engine_sha256"] == sha256_file(out / ENGINE_FILE)
    validate_engine(record, input_size=352, source_sha256=sha256_file(weights))


def test_the_fp16_flag_reaches_the_builder_and_fp32_does_not(tmp_path, monkeypatch):
    """The one branch that decides the engine's precision is unpinned: nothing asserts it.

    A fake ``trt`` records the flags the builder config is given, so dropping the
    ``set_flag(FP16)`` line and making it unconditional both show up here. TF32 is cleared
    in both arms (see ``test_the_builder_clears_tf32_before_it_builds``).
    """
    flags = []
    cleared = []

    class _Config:
        def set_memory_pool_limit(self, *args):
            pass

        def set_flag(self, flag):
            flags.append(flag)

        def clear_flag(self, flag):
            cleared.append(flag)

    class _Builder:
        def __init__(self, logger):
            self._config = _Config()

        def create_network(self):
            return object()

        def create_builder_config(self):
            return self._config

        def build_serialized_network(self, network, config):
            return b"serialized-engine"

    class _Parser:
        def __init__(self, network, logger):
            pass

        def parse(self, payload):
            return True

    class _Logger:
        WARNING = 0

        def __init__(self, level=None):
            pass

    fake_trt = types.SimpleNamespace(
        Logger=_Logger, Builder=_Builder, OnnxParser=_Parser,
        BuilderFlag=types.SimpleNamespace(FP16="FP16", TF32="TF32"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="WORKSPACE"),
        __version__="10.3.0")
    monkeypatch.setattr(export_trt, "_import_trt", lambda: fake_trt)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx-graph")

    engine_bytes, version = export_trt.build_engine(onnx_path, fp16=True, workspace_mb=64)
    assert engine_bytes == b"serialized-engine" and version == "10.3.0"
    assert flags == ["FP16"] and cleared == ["TF32"]

    flags.clear()
    cleared.clear()
    export_trt.build_engine(onnx_path, fp16=False, workspace_mb=64)
    assert flags == [] and cleared == ["TF32"]


def test_the_builder_clears_tf32_before_it_builds(tmp_path, monkeypatch):
    """TensorRT picks TF32 tactics for an fp32 network unless the config refuses them.

    TF32 computes fp32 operands with a 10-bit mantissa, so an engine the record calls fp32
    would not be fp32. The numeric effect is not observable off the box, so this pins the
    call -- and its *order*: the clear has to be on the config the build is handed, not a
    later no-op. A fake trt logs every call on the config and the build itself, in one
    sequence, so moving the clear after ``build_serialized_network`` fails here.
    """
    calls = []

    class _Config:
        def set_memory_pool_limit(self, *args):
            calls.append(("set_memory_pool_limit",) + args)

        def set_flag(self, flag):
            calls.append(("set_flag", flag))

        def clear_flag(self, flag):
            calls.append(("clear_flag", flag))

    class _Builder:
        def __init__(self, logger):
            self._config = _Config()

        def create_network(self):
            return object()

        def create_builder_config(self):
            return self._config

        def build_serialized_network(self, network, config):
            calls.append(("build", config))
            return b"serialized-engine"

    class _Parser:
        def __init__(self, network, logger):
            pass

        def parse(self, payload):
            return True

    class _Logger:
        WARNING = 0

        def __init__(self, level=None):
            pass

    fake_trt = types.SimpleNamespace(
        Logger=_Logger, Builder=_Builder, OnnxParser=_Parser,
        BuilderFlag=types.SimpleNamespace(FP16="FP16", TF32="TF32"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="WORKSPACE"),
        __version__="10.3.0")
    monkeypatch.setattr(export_trt, "_import_trt", lambda: fake_trt)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx-graph")
    export_trt.build_engine(onnx_path, fp16=False, workspace_mb=64)

    assert ("clear_flag", "TF32") in calls
    assert calls.index(("clear_flag", "TF32")) < _build_index(calls)

    # The clear is unconditional: an fp16 build keeps it, beside the FP16 flag.
    calls.clear()
    export_trt.build_engine(onnx_path, fp16=True, workspace_mb=64)
    assert calls.index(("clear_flag", "TF32")) < _build_index(calls)
    assert calls.index(("set_flag", "FP16")) < _build_index(calls)


def _build_index(calls):
    """Index of the ``("build", config)`` entry in a call log from the fake above."""
    return next(i for i, call in enumerate(calls) if call[0] == "build")


def test_the_export_refuses_fp16_for_this_graph(tmp_path, monkeypatch, capsys):
    """An fp16 engine of this model is non-finite, so the export refuses the flag.

    The refusal has to come before the checkpoint is read and before the builder is asked
    for an engine, so both seams raise if they are reached. ``build_engine`` itself keeps
    its flag -- the test above pins that branch.
    """
    _deps_present(monkeypatch)

    def _unreachable(*args, **kwargs):
        raise AssertionError("the export must refuse --fp16 before it does any work")

    monkeypatch.setattr(export_trt, "_load_checkpoint", _unreachable)
    monkeypatch.setattr(export_trt, "build_engine", _unreachable)

    out = tmp_path / "out"
    assert export_trt.main([str(_checkpoint(tmp_path)), "--out", str(out), "--fp16"]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "--fp16" in err and "non-finite" in err
    assert not out.exists()


def _stub_export_seams(monkeypatch, checkpoint):
    """Stub the expensive halves of a successful export, so a test can reach --verify."""
    monkeypatch.setattr(
        export_trt, "_load_checkpoint",
        lambda path: (types.SimpleNamespace(decoder_arch="v1"), checkpoint / "model.pth", 352))
    monkeypatch.setattr(
        export_trt, "export_onnx",
        lambda model, input_size, onnx_path, opset: onnx_path.write_bytes(b"onnx-graph"))
    monkeypatch.setattr(
        export_trt, "build_engine",
        lambda onnx_path, fp16, workspace_mb: (b"serialized-engine", "10.3.0"))
    monkeypatch.setattr(
        export_trt, "engine_tensors",
        lambda engine_bytes, input_name, input_size: (
            "float32", [{"name": "logits", "shape": [1, 2, 22, 22], "dtype": "float32"}]))


def test_a_refused_verify_leaves_no_output_directory(tmp_path, monkeypatch, capsys):
    """A --verify that refuses used to leave model.onnx and model.engine under --out.

    With no engine.json beside them, the overwrite check then refused every retry, so one
    bad verdict locked the directory out until someone deleted it by hand.
    """
    _deps_present(monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    _stub_export_seams(monkeypatch, checkpoint)

    def _refuse(*args, **kwargs):
        raise EngineError("max abs logit diff 3.5 is over 0.25")

    monkeypatch.setattr(export_trt, "verify", _refuse)

    out = tmp_path / "out"
    assert export_trt.main([str(checkpoint), "--out", str(out), "--verify"]) == 2
    assert "error:" in capsys.readouterr().err
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt"]


def test_a_verify_that_cannot_run_off_cuda_leaves_no_output_directory(
        tmp_path, monkeypatch, capsys):
    """The same refusal through the real ``verify``, not a stubbed one: it needs CUDA."""
    _deps_present(monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    _stub_export_seams(monkeypatch, checkpoint)
    monkeypatch.setattr(export_trt, "load_engine_from", lambda engine_path, record: object())

    out = tmp_path / "out"
    rc = export_trt.main([str(checkpoint), "--out", str(out), "--verify", "--device", "cpu"])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt"]


def test_the_output_directory_has_the_mode_mkdir_would_give_it(tmp_path, monkeypatch):
    """Staging must not change --out's permissions.

    ``mkdtemp`` creates its directory 0700 and ``rename`` moves the inode, so the staged
    export handed --out the staging directory's owner-only mode where the plain ``mkdir``
    it replaced gave 0o777 minus the umask. These are files the demos and other tools
    read. The reference directory is what ``mkdir`` produces under the same umask.
    """
    _deps_present(monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    _stub_export_seams(monkeypatch, checkpoint)

    previous = os.umask(0o022)
    try:
        reference = tmp_path / "reference"
        reference.mkdir()
        out = tmp_path / "out"
        assert export_trt.main([str(checkpoint), "--out", str(out)]) == 0
        expected = stat.S_IMODE(reference.stat().st_mode)
        actual = stat.S_IMODE(out.stat().st_mode)
    finally:
        os.umask(previous)

    assert expected == 0o755  # the umask pinned above is really in force
    assert actual == expected


@pytest.mark.skipif(not _has_tensorrt(), reason="TensorRT is not installed on this host")
def test_tensorrt_is_importable_when_the_host_has_it():
    assert trt_engine._import_trt().__version__


# --------------------------------------------------------------------------
# The bench
# --------------------------------------------------------------------------
def test_the_bench_takes_an_engine_flag_and_defaults_to_none():
    assert bench_model._parse_args([]).engine is None
    assert bench_model._parse_args(["--engine", "/e"]).engine == Path("/e")


def test_the_bench_runs_the_torch_model_without_an_engine():
    args = bench_model._parse_args(
        ["--arch", "v1", "--device", "cpu", "--input-size", "64", "--warmup", "0", "--iters", "2"])
    assert bench_model._load_engine(args, 64, torch.device("cpu")) is None
    report = bench_model.run(args)
    assert report["backend"] == "torch" and report["dtype"] == "float32"


def test_the_bench_refuses_an_engine_off_cuda():
    args = bench_model._parse_args(["--arch", "v1", "--engine", "/e", "--device", "cpu"])
    with pytest.raises(bench_model.BenchError, match="CUDA"):
        bench_model._load_engine(args, 352, torch.device("cpu"))


def test_the_bench_refuses_fp16_together_with_an_engine():
    args = bench_model._parse_args(
        ["--arch", "v1", "--engine", "/e", "--fp16", "--device", "cuda"])
    with pytest.raises(bench_model.BenchError, match="fp16"):
        bench_model._load_engine(args, 352, torch.device("cuda"))


def test_the_bench_refuses_an_engine_without_a_checkpoint_to_validate_against():
    args = bench_model._parse_args(["--arch", "v1", "--engine", "/e", "--device", "cuda"])
    with pytest.raises(bench_model.BenchError, match="--checkpoint"):
        bench_model._load_engine(args, 352, torch.device("cuda"))


def test_the_bench_refuses_an_engine_that_does_not_match_the_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.pth").write_bytes(b"weights")
    directory = _engine_dir(tmp_path, input_size=192)
    args = bench_model._parse_args(
        ["--arch", "v1", "--engine", str(directory), "--device", "cuda", "--checkpoint", str(ckpt)])
    with pytest.raises(bench_model.BenchError, match="192"):
        bench_model._load_engine(args, 352, torch.device("cuda"))


# --------------------------------------------------------------------------
# The realtime demo (read as text: it needs cv2, which this VM has not)
# --------------------------------------------------------------------------
def test_the_realtime_demo_takes_an_engine_flag():
    assert '"--engine"' in REALTIME_SRC


def test_the_realtime_demo_runs_the_engine_only_when_it_is_given_one():
    assert "if engine is not None:" in REALTIME_SRC
    assert "logits = engine.infer(x)" in REALTIME_SRC
    # The torch forward stays for the no-engine case.
    assert "logits = demo_forward(model, x, fp16=args.fp16)" in REALTIME_SRC


def test_the_realtime_demo_validates_the_engine_against_the_checkpoint_it_serves():
    assert "TrtEngine.load(" in REALTIME_SRC
    assert "source_sha256=sha256_file(WEIGHTS)" in REALTIME_SRC
