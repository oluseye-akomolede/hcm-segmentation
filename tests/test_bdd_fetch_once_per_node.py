"""The BDD prepared tree is downloaded by one process per node, not by every rank.

The incident: a 2-GPU torchrun job had both ranks call `fetch_prepared_tree` on
the same `/tmp` directory. MinIO writes each object to `<file>.<hash>.part` and
renames it, so the ranks raced; rank 1's rename raised FileNotFoundError and
torchrun killed rank 0 a minute and a half in.

These tests pin the call path, not the download: the order of the fetch and the
barrier for each rank, and that a single-process run never barriers.
"""

from pathlib import Path

from ecs_hcm_model.core import trainer as T
from ecs_hcm_model.core.trainer import DistributedContext


class _FakeDist:
    """Records the order of the barrier against the fetches the test records."""

    def __init__(self, events, initialized=True):
        self.events = events
        self._initialized = initialized

    def is_available(self):
        return True

    def is_initialized(self):
        return self._initialized

    def barrier(self):
        self.events.append("barrier")


def _install(monkeypatch, events, initialized=True):
    monkeypatch.setattr(T, "dist", _FakeDist(events, initialized))
    monkeypatch.setattr(
        T, "fetch_prepared_tree",
        lambda root: events.append("fetch") or Path(root),
    )


def test_the_nodes_first_process_fetches_before_the_barrier(monkeypatch):
    events = []
    _install(monkeypatch, events)
    ctx = DistributedContext(rank=0, local_rank=0, world_size=2)

    result = T.fetch_prepared_tree_once_per_node(ctx, "/tmp/bdd")

    assert events == ["fetch", "barrier"]
    assert result == Path("/tmp/bdd")


def test_a_nodes_other_process_never_fetches_before_the_barrier(monkeypatch):
    """The regression: this is the rank that raced rank 0 on the shared /tmp."""
    events = []
    _install(monkeypatch, events)
    ctx = DistributedContext(rank=1, local_rank=1, world_size=2)

    result = T.fetch_prepared_tree_once_per_node(ctx, "/tmp/bdd")

    assert events == ["barrier", "fetch"]
    assert result == Path("/tmp/bdd")


def test_the_gate_is_the_local_rank_not_the_global_one(monkeypatch):
    """A second node's first GPU has a global rank above 0 and must still fetch.

    Each node fetches into its own /tmp; gating on the global rank would leave
    every node but the first with no download at all.
    """
    events = []
    _install(monkeypatch, events)
    ctx = DistributedContext(rank=2, local_rank=0, world_size=4)

    T.fetch_prepared_tree_once_per_node(ctx, "/tmp/bdd")

    assert events == ["fetch", "barrier"]


def test_a_single_process_run_takes_the_plain_path(monkeypatch):
    events = []
    _install(monkeypatch, events)
    ctx = DistributedContext(rank=0, local_rank=0, world_size=1)

    result = T.fetch_prepared_tree_once_per_node(ctx, "/tmp/bdd")

    assert events == ["fetch"]
    assert result == Path("/tmp/bdd")


def test_no_process_group_means_no_barrier(monkeypatch):
    """A world size above one with no group up would hang at the barrier."""
    events = []
    _install(monkeypatch, events, initialized=False)
    ctx = DistributedContext(rank=1, local_rank=1, world_size=2)

    T.fetch_prepared_tree_once_per_node(ctx, "/tmp/bdd")

    assert events == ["fetch"]
