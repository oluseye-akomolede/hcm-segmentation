"""Environment knobs applied at trainer startup.

Covers the OMP_NUM_THREADS -> torch.set_num_threads bridge in
``set_env_vars``. These run on CPU with no MLflow and no network: they only
assert on torch's own thread count and on what the function does with a bad
value.
"""

import pytest
import torch

from ecs_hcm_model.core.trainer import set_env_vars


@pytest.fixture
def thread_restore():
    """Put torch's thread count back however the test left it."""
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


class TestOmpThreadLimit:
    def test_omp_num_threads_is_applied_to_torch(self, monkeypatch, thread_restore):
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        set_env_vars()
        assert torch.get_num_threads() == 3

    def test_absent_omp_num_threads_leaves_torch_alone(
        self, monkeypatch, thread_restore
    ):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        before = torch.get_num_threads()
        set_env_vars()
        assert torch.get_num_threads() == before

    def test_empty_omp_num_threads_leaves_torch_alone(
        self, monkeypatch, thread_restore
    ):
        # An exported-but-empty variable is a common shell accident; it must
        # not be parsed as 0 and silently collapse torch to one thread.
        monkeypatch.setenv("OMP_NUM_THREADS", "")
        before = torch.get_num_threads()
        set_env_vars()
        assert torch.get_num_threads() == before

    def test_unparseable_value_is_ignored_not_fatal(
        self, monkeypatch, thread_restore, capsys
    ):
        monkeypatch.setenv("OMP_NUM_THREADS", "many")
        before = torch.get_num_threads()
        set_env_vars()  # must not raise
        assert torch.get_num_threads() == before
        assert "OMP_NUM_THREADS" in capsys.readouterr().out

    @pytest.mark.parametrize("bad", ["0", "-4"])
    def test_nonpositive_value_is_ignored(
        self, monkeypatch, thread_restore, capsys, bad
    ):
        monkeypatch.setenv("OMP_NUM_THREADS", bad)
        before = torch.get_num_threads()
        set_env_vars()  # must not raise
        assert torch.get_num_threads() == before
        assert "OMP_NUM_THREADS" in capsys.readouterr().out

    def test_each_rank_takes_its_share_of_the_node_thread_budget(
        self, monkeypatch, thread_restore
    ):
        # OMP_NUM_THREADS describes the node (or the pod), and the ranks on that
        # node are separate processes each free to start that many threads, so
        # two ranks left alone would run at twice the node's budget.
        monkeypatch.setenv("OMP_NUM_THREADS", "12")
        set_env_vars(world_size=2)
        assert torch.get_num_threads() == 6

    def test_a_rank_share_never_collapses_to_zero(self, monkeypatch, thread_restore):
        # A budget smaller than the world size must floor at one thread, not
        # divide down to a torch.set_num_threads(0).
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        set_env_vars(world_size=4)
        assert torch.get_num_threads() == 1

    def test_set_env_vars_still_sets_the_other_variables(self, monkeypatch):
        import os

        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        set_env_vars()
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "500"
