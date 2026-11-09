"""How the demo decides which resolution to feed the model.

``realtime_demo.py`` used to hardcode ``INPUT_W, INPUT_H = 192, 192``. A model
trained at another size was still fed 192-pixel frames -- and because a
convolutional network accepts any input size, the demo kept running and drawing
an overlay computed at a resolution the model never saw. Nothing looked wrong.

The size the trainer used is now recorded in the model's own metadata, so the
demo can read it instead of assuming: ``save_epoch_checkpoint`` writes
``input_size`` into the metadata that ``upload_model`` nests under
``custom_metadata``. Models published before that field existed are 192-pixel
models, so 192 remains the default.

This module holds the rule on its own: no ``cv2``, no torch, and no work at
import, so the rule can be tested without a camera, a GPU or a model -- and so
a test that imports it cannot open the Jetson's camera by accident. The same
reasoning is why ``frame_stream.py`` is shaped this way.

A value that is *stated* but unusable is an error rather than a fall-through.
Falling back to the default on a garbled size would put the demo back in the
state this module exists to remove: running, and wrong.
"""

import json
from collections.abc import Mapping

DEFAULT_INPUT_SIZE = 192
"""The size of models published before a checkpoint recorded its own."""

ENV_VAR = "ECS_HCM_DEMO_INPUT_SIZE"

SOURCE_ENV = f"env:{ENV_VAR}"
SOURCE_CUSTOM_METADATA = "metadata.custom_metadata.input_size"
SOURCE_TOP_LEVEL = "metadata.input_size"
SOURCE_DEFAULT = "default"


class InputSizeError(ValueError):
    """A stated input size that cannot be used as a positive whole number.

    Also raised for a ``metadata.json`` that exists but cannot be read: the
    size it states is then unknown rather than absent.
    """


def load_metadata(path):
    """Parse the ``metadata.json`` sidecar at ``path``, or ``None`` if absent.

    A missing sidecar is legal (models predating it are 192-pixel models). A
    sidecar that is present but unreadable, not valid JSON, or not a JSON
    object raises :class:`InputSizeError` naming the file and the error: the
    demo used to print a line and carry on at 192, so a truncated copy of a
    352 model's metadata ran it at the wrong size with nothing visibly wrong.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise InputSizeError(f"cannot read {path}: {exc}") from exc
    try:
        meta = json.loads(text)
    except ValueError as exc:
        raise InputSizeError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(meta, Mapping):
        raise InputSizeError(
            f"{path} holds a JSON {type(meta).__name__}, not an object"
        )
    return meta


def _as_positive_int(value):
    """Return ``value`` as a positive ``int``, or ``None`` when it is unstated.

    Raised as :class:`InputSizeError` when the value is stated but unusable.

    ``None`` and the empty string mean *unstated* rather than *wrong*: JSON
    writers emit an explicit ``null`` for "this was not recorded", a shell
    variable can be set to nothing by an unset expansion, and both describe the
    same situation as the key being absent -- so they fall through to the next
    source. ``True`` is rejected even though Python counts it as an ``int``: a
    boolean in a size field is a corrupt artifact, not a size of 1.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        raise InputSizeError(f"{value!r} is not a whole number of pixels")

    if isinstance(value, int):
        size = value
    elif isinstance(value, float):
        # JSON has one number type, so a size that round-tripped through a
        # writer that emits 352.0 is still the size 352.
        if not value.is_integer():
            raise InputSizeError(f"{value!r} is not a whole number of pixels")
        size = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            size = int(text, 10)
        except ValueError as exc:
            raise InputSizeError(f"{value!r} is not a whole number of pixels") from exc
    else:
        raise InputSizeError(f"{value!r} is not a whole number of pixels")

    if size <= 0:
        raise InputSizeError(f"{size} is not a positive number of pixels")
    return size


def resolve_input_size(meta, env):
    """Resolve the demo's input size and report where it came from.

    ``meta`` is the parsed ``metadata.json`` sidecar, or ``None`` when there is
    none; ``env`` is a mapping of environment variables (``os.environ`` in the
    demo, a plain dict in tests). Returns ``(size, source)``.

    The order is the one the change specifies: the environment, then
    ``custom_metadata.input_size``, then a top-level ``input_size``, then the
    default. The environment comes first because it is the operator's escape
    hatch on a device whose artifact directory may hold an older model.
    """
    # A sidecar that parsed to something other than an object is corrupt, not
    # absent: falling back to 192 here would run a 352 model at the wrong size.
    if meta is None:
        meta = {}
    elif not isinstance(meta, Mapping):
        raise InputSizeError(
            f"the metadata is a {type(meta).__name__}, not an object"
        )

    if ENV_VAR in env:
        size = _as_positive_int(env[ENV_VAR])
        if size is not None:
            return size, SOURCE_ENV

    custom = meta.get("custom_metadata")
    if isinstance(custom, Mapping) and "input_size" in custom:
        size = _as_positive_int(custom["input_size"])
        if size is not None:
            return size, SOURCE_CUSTOM_METADATA

    if "input_size" in meta:
        size = _as_positive_int(meta["input_size"])
        if size is not None:
            return size, SOURCE_TOP_LEVEL

    return DEFAULT_INPUT_SIZE, SOURCE_DEFAULT


SOURCE_CLI = "--input-size"


def resolve_with_cli(cli_size, meta, env):
    """Like :func:`resolve_input_size`, but an explicit ``--input-size`` wins.

    For demos that take the size on the command line (``video/image_demo.py``):
    ``cli_size`` is the parsed flag, ``None`` when it was not given. A given
    flag is checked like any stated size; otherwise the usual order applies.
    """
    if cli_size is not None:
        size = _as_positive_int(cli_size)
        if size is None:
            raise InputSizeError(f"{cli_size!r} is not a whole number of pixels")
        return size, SOURCE_CLI
    return resolve_input_size(meta, env)
