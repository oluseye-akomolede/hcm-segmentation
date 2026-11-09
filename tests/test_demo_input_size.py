"""Which resolution the demo feeds the model, and how it decides.

These tests import ``jetson.input_size`` only. That module has no ``cv2``, no
torch and no work at import -- and this VM has no ``cv2`` installed at all, so
the import succeeding here is itself the check that the rule is reachable
without a camera or a GPU. ``realtime_demo.py`` cannot be imported in a test
(importing it binds port 5000 and loads weights), so the last class reads it as
text, the way ``test_demo_stream.py`` guards the MJPEG boundary.

The regression: the demo hardcoded ``INPUT_W, INPUT_H = 192, 192``. A model
trained at 352 was still fed 192-pixel frames, and because a convolutional
network accepts any input size the demo kept running and drew a plausible
overlay computed at a resolution the model never saw. Nothing failed; nothing
looked wrong.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetson.input_size import (  # noqa: E402
    DEFAULT_INPUT_SIZE,
    ENV_VAR,
    SOURCE_CLI,
    SOURCE_CUSTOM_METADATA,
    SOURCE_DEFAULT,
    SOURCE_ENV,
    SOURCE_TOP_LEVEL,
    InputSizeError,
    load_metadata,
    resolve_input_size,
    resolve_with_cli,
)


_SENTINEL = object()


def _meta(custom=_SENTINEL, top=_SENTINEL):
    """A sidecar shaped like the one ``upload_model`` writes.

    A parameter left at the sentinel means the key is *absent*, which is
    different from passing ``None`` (the key present with a null value).
    """
    meta = {"name": "hcm_coco_seg", "architecture": "hcm"}
    if custom is not _SENTINEL:
        meta["custom_metadata"] = custom
    if top is not _SENTINEL:
        meta["input_size"] = top
    return meta


class TestTheEnvironmentWins:
    def test_the_environment_overrides_the_models_metadata(self):
        meta = _meta(custom={"input_size": 352})
        assert resolve_input_size(meta, {ENV_VAR: "512"}) == (512, SOURCE_ENV)

    def test_the_environment_is_used_when_there_is_no_metadata(self):
        assert resolve_input_size(None, {ENV_VAR: "352"}) == (352, SOURCE_ENV)

    def test_an_absent_environment_variable_falls_through(self):
        meta = _meta(custom={"input_size": 352})
        assert resolve_input_size(meta, {}) == (352, SOURCE_CUSTOM_METADATA)

    def test_the_variable_is_named_as_the_operator_documented(self):
        # The launch script and the runbook spell this name; a rename would
        # silently stop the override from applying.
        assert ENV_VAR == "ECS_HCM_DEMO_INPUT_SIZE"
        assert SOURCE_ENV.endswith(ENV_VAR)


class TestTheModelsOwnMetadata:
    def test_custom_metadata_supplies_the_size(self):
        assert resolve_input_size(_meta(custom={"input_size": 352}), {}) == (
            352,
            SOURCE_CUSTOM_METADATA,
        )

    def test_custom_metadata_wins_over_the_top_level_entry(self):
        meta = _meta(custom={"input_size": 352}, top=192)
        assert resolve_input_size(meta, {}) == (352, SOURCE_CUSTOM_METADATA)

    def test_the_top_level_entry_is_read_when_custom_metadata_is_absent(self):
        assert resolve_input_size(_meta(top=352), {}) == (352, SOURCE_TOP_LEVEL)

    def test_the_top_level_entry_is_read_when_custom_metadata_has_no_size(self):
        meta = _meta(custom={"task": "coco_seg"}, top=352)
        assert resolve_input_size(meta, {}) == (352, SOURCE_TOP_LEVEL)

    def test_custom_metadata_that_is_not_a_mapping_is_skipped(self):
        # ``"input_size" in "input_size"`` is a substring test that would be
        # true here, so the shape has to be checked rather than the containment.
        meta = _meta(custom="input_size", top=352)
        assert resolve_input_size(meta, {}) == (352, SOURCE_TOP_LEVEL)

    def test_the_size_the_trainer_writes_is_what_this_reads(self):
        # save_epoch_checkpoint puts input_size in the metadata dict that
        # upload_model nests under custom_metadata (trainer.py:773, :1971).
        meta = _meta(custom={"input_size": 352, "task": "coco_seg"})
        assert resolve_input_size(meta, {})[0] == 352


class TestTheDefault:
    def test_no_metadata_and_no_environment_is_192(self):
        assert resolve_input_size(None, {}) == (DEFAULT_INPUT_SIZE, SOURCE_DEFAULT)
        assert DEFAULT_INPUT_SIZE == 192

    def test_an_empty_sidecar_is_192(self):
        assert resolve_input_size({}, {}) == (192, SOURCE_DEFAULT)

    @pytest.mark.parametrize("not_a_mapping", [[], [352], "metadata", 7])
    def test_a_sidecar_that_is_not_a_mapping_is_corrupt(self, not_a_mapping):
        # A sidecar that parsed to the wrong shape is corrupt, not absent:
        # treating it as 192 would run a 352 model at the wrong size.
        with pytest.raises(InputSizeError, match="not an object"):
            resolve_input_size(not_a_mapping, {})

    def test_the_cli_path_refuses_it_too(self):
        with pytest.raises(InputSizeError, match="not an object"):
            resolve_with_cli(None, [352], {})

    def test_a_sidecar_without_the_key_is_192(self):
        # The 09-29 COCO weights: published before the size was recorded.
        meta = _meta(custom={"task": "coco_seg", "epoch": 3})
        assert resolve_input_size(meta, {}) == (192, SOURCE_DEFAULT)

    def test_the_default_matches_the_trainer_default(self):
        # An older model has no recorded size, so the demo's fallback has to be
        # the size that trainer defaults to -- otherwise it recognises nothing.
        from ecs_hcm_model.core.model_utils import DEFAULT_INPUT_SIZE as TRAINER_DEFAULT

        assert DEFAULT_INPUT_SIZE == TRAINER_DEFAULT


class TestAStatedSizeThatCannotBeUsed:
    """Stated and unusable is an error; it must not become the default.

    Falling back here would put the demo back in the state the change removes:
    running, and drawing an overlay at a resolution the model never saw.
    """

    @pytest.mark.parametrize("bad", ["abc", "0", "-8", "1.5", "192px", "true", "None"])
    def test_a_bad_environment_value_raises(self, bad):
        with pytest.raises(InputSizeError):
            resolve_input_size(None, {ENV_VAR: bad})

    def test_a_bad_environment_value_does_not_fall_through_to_good_metadata(self):
        meta = _meta(custom={"input_size": 352})
        with pytest.raises(InputSizeError):
            resolve_input_size(meta, {ENV_VAR: "abc"})

    @pytest.mark.parametrize("bad", [0, -352, True, False, "abc", 352.5, ["352"]])
    def test_a_bad_metadata_value_raises(self, bad):
        with pytest.raises(InputSizeError):
            resolve_input_size(_meta(custom={"input_size": bad}), {})

    def test_a_bad_custom_metadata_value_does_not_fall_through_to_the_top_level(self):
        meta = _meta(custom={"input_size": "abc"}, top=352)
        with pytest.raises(InputSizeError):
            resolve_input_size(meta, {})

    def test_a_bad_top_level_value_raises(self):
        with pytest.raises(InputSizeError):
            resolve_input_size(_meta(top=0), {})

    def test_the_error_names_the_value_it_rejected(self):
        with pytest.raises(InputSizeError, match="abc"):
            resolve_input_size(None, {ENV_VAR: "abc"})

    @pytest.mark.parametrize("unstated", ["", "   ", None])
    def test_an_empty_environment_value_is_unstated_not_wrong(self, unstated):
        # An unset expansion in a shell script leaves an empty variable; that
        # is "not set", not "set to something unusable".
        meta = _meta(custom={"input_size": 352})
        assert resolve_input_size(meta, {ENV_VAR: unstated}) == (
            352,
            SOURCE_CUSTOM_METADATA,
        )

    def test_a_null_in_the_sidecar_is_unstated_not_wrong(self):
        # JSON writers emit an explicit null for "not recorded".
        meta = _meta(custom={"input_size": None}, top=352)
        assert resolve_input_size(meta, {}) == (352, SOURCE_TOP_LEVEL)


class TestTheValueForms:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (352, 352),
            ("352", 352),
            (" 352 ", 352),
            (352.0, 352),
        ],
    )
    def test_a_usable_size_is_read_from_each_form(self, value, expected):
        assert resolve_input_size(_meta(custom={"input_size": value}), {})[0] == expected

    def test_the_size_is_returned_as_an_int(self):
        size, _source = resolve_input_size(_meta(custom={"input_size": "352"}), {})
        assert isinstance(size, int)
        assert not isinstance(size, bool)


@pytest.fixture(params=["jetson/realtime_demo.py", "video/video_demo.py"])
def demo_path(request):
    return REPO_ROOT / request.param


class TestTheSidecarIsReadOrRefused:
    """A missing metadata.json is legal; a present but unreadable one stops."""

    def test_a_missing_sidecar_is_none(self, tmp_path):
        assert load_metadata(tmp_path / "metadata.json") is None

    def test_a_good_sidecar_is_parsed(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text('{"custom_metadata": {"input_size": 352}}')
        meta = load_metadata(path)
        assert resolve_input_size(meta, {}) == (352, SOURCE_CUSTOM_METADATA)

    @pytest.mark.parametrize(
        "text",
        [
            '{"custom_metadata": {"input_size": 35',  # truncated copy
            "",  # zero-length file
            "not json at all",
        ],
    )
    def test_an_unparseable_sidecar_raises_naming_the_file(self, tmp_path, text):
        path = tmp_path / "metadata.json"
        path.write_text(text)
        with pytest.raises(InputSizeError, match="metadata.json"):
            load_metadata(path)

    @pytest.mark.parametrize("text", ["[352]", "352", '"352"', "null"])
    def test_a_sidecar_that_is_not_an_object_raises(self, tmp_path, text):
        path = tmp_path / "metadata.json"
        path.write_text(text)
        with pytest.raises(InputSizeError, match="not an object"):
            load_metadata(path)

    def test_a_sidecar_that_is_not_utf8_raises(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_bytes(b"\xff\xfe\x00garbage")
        with pytest.raises(InputSizeError, match="metadata.json"):
            load_metadata(path)

    def test_a_sidecar_that_is_a_directory_raises(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.mkdir()
        with pytest.raises(InputSizeError, match="cannot read"):
            load_metadata(path)


class TestTheDemoUsesIt:
    """Read as text: importing a demo opens a camera or frames dir and loads a model.

    Both demos are checked: the offline video demo hardcoded the same 192 pair.
    """

    @pytest.fixture(autouse=True)
    def _bind(self, demo_path):
        self._path = demo_path

    def _src(self):
        return self._path.read_text()

    def test_the_demo_reuses_the_resolver_rather_than_a_copy(self):
        assert (
            "from jetson.input_size import InputSizeError, load_metadata, resolve_input_size"
            in self._src()
        )
        assert "def resolve_input_size" not in self._src()
        assert "def load_metadata" not in self._src()

    def test_the_hardcoded_pair_is_gone(self):
        src = self._src()
        assert "INPUT_W" not in src and "INPUT_H" not in src, (
            "the fixed input size is back; a 352 model would be fed 192 frames "
            "and still look like it was working"
        )

    def test_the_demo_resolves_the_size_from_the_metadata_and_the_environment(self):
        src = self._src()
        assert "resolve_input_size(meta, os.environ)" in src
        assert "input_size" in src

    def test_the_resolved_size_reaches_every_use(self):
        src = self._src()
        # the frame handed to the model
        assert "preprocess_bgr(frame, input_size)" in src
        # the warmup tensor
        assert "torch.zeros(1, 3, input_size, input_size" in src
        # the logits upsample
        assert "size=(input_size, input_size)" in src

    def test_the_resize_is_the_whole_frame_and_not_a_crop(self):
        src = self._src()
        assert "cv2.resize(rgb, (input_size, input_size)" in src

    def test_the_chosen_size_and_source_are_printed(self):
        # The shared loader prints them, from what the demo hands it.
        assert "input_size=input_size, input_size_source=input_size_source)" in self._src()
        loader = (REPO_ROOT / "jetson" / "demo_model.py").read_text()
        assert "input size: {input_size} (from {input_size_source})" in loader

    def test_a_bad_size_stops_the_demo_instead_of_continuing(self):
        src = self._src()
        assert "InputSizeError" in src
        assert "SystemExit" in src

    def test_an_unreadable_sidecar_is_no_longer_swallowed(self):
        src = self._src()
        assert "(continuing)" not in src, "the corrupt-sidecar fall-through is back"
        # The load sits inside the same try that turns InputSizeError into
        # SystemExit, so a corrupt sidecar stops the demo like a garbled size.
        # The try that load_metadata belongs to is the last one opened before
        # its except InputSizeError (other try blocks may come earlier).
        try_block = src.split("except InputSizeError")[0].rsplit("try:", 1)[1]
        assert "load_metadata(META)" in try_block

    def test_the_resolver_module_stays_free_of_cv2_and_torch(self):
        # Importability on this VM (no cv2 installed) is the real check; this
        # guards against the imports creeping back in as a convenience.
        src = (REPO_ROOT / "jetson" / "input_size.py").read_text()
        assert "import cv2" not in src
        assert "import torch" not in src


class TestTheImageDemoTakesTheFlagThenTheModel:
    """video/image_demo.py defaulted --input-size to 192 and ignored metadata."""

    def test_a_given_flag_wins_over_metadata_and_environment(self):
        meta = _meta(custom={"input_size": 352})
        assert resolve_with_cli(288, meta, {ENV_VAR: "512"}) == (288, SOURCE_CLI)

    def test_no_flag_reads_the_model_metadata(self):
        meta = _meta(custom={"input_size": 352})
        assert resolve_with_cli(None, meta, {}) == (352, SOURCE_CUSTOM_METADATA)

    def test_no_flag_keeps_the_resolver_order(self):
        meta = _meta(custom={"input_size": 352})
        assert resolve_with_cli(None, meta, {ENV_VAR: "512"}) == (512, SOURCE_ENV)
        assert resolve_with_cli(None, None, {}) == (192, SOURCE_DEFAULT)

    @pytest.mark.parametrize("bad", [0, -192])
    def test_an_unusable_flag_raises(self, bad):
        with pytest.raises(InputSizeError):
            resolve_with_cli(bad, None, {})

    def test_the_demo_wires_it(self):
        src = (REPO_ROOT / "video" / "image_demo.py").read_text()
        assert "from jetson.input_size import InputSizeError, load_metadata, resolve_with_cli" in src
        assert 'default=192' not in src
        assert '"--input-size", type=int, default=None' in src
        try_block = src.split("except InputSizeError")[0].rsplit("try:", 1)[1]
        assert 'load_metadata(weights.with_name("metadata.json"))' in try_block
        assert "resolve_with_cli(args.input_size, meta, os.environ)" in try_block
        assert "SystemExit" in src
        assert "input_size=input_hw, input_size_source=input_size_source)" in src
        for use in ("load_model(weights, device, input_size, meta,",
                    "preprocess_bgr(frame, input_size)",
                    "overlay_heat_and_contours(frame, logits, input_size)"):
            assert use in src
