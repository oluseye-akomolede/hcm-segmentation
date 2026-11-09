"""The worker-aware split, driven by a real DataLoader instead of a faked worker.

tests/test_local_dataset_sharding.py replaces GeneratorDataset's module-level
``get_worker_info`` with a SimpleNamespace, so the identity the split is computed from is
typed by the test. This module does not patch it: two ranks are built in this process, each
wrapped in a DataLoader of two fork workers, and the split is read from what those worker
processes actually yielded.
"""

import functools

import torch

from ecs_hcm_model.core.GeneratorDataset import GeneratorDataset


def _int_items(n, shards, s3):
    """A stand-in for the S3 arrow stream: the first ``n`` integers, in order.

    ``__iter__`` calls ``self.arrow_file_generator(shards=..., s3=...)``. A module-level
    function behind ``functools.partial`` keeps the assigned attribute picklable, so the
    test does not depend on the loader forking rather than spawning.
    """
    return iter(range(n))


def test_two_ranks_with_real_dataloader_workers_are_disjoint_and_complete(monkeypatch):
    stream_length = 40
    num_workers = 2
    world_size = 2

    monkeypatch.setenv("MINIO_ACCESS_KEY", "dummy")
    monkeypatch.setenv("MINIO_SECRET_KEY", "dummy")

    rank_items = []
    for rank in range(world_size):
        ds = GeneratorDataset(rank=rank, world_size=world_size)
        ds.arrow_file_generator = functools.partial(_int_items, stream_length)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=num_workers)
        rank_items.append([int(batch[0]) for batch in loader])

    all_items = rank_items[0] + rank_items[1]

    assert set(rank_items[0]).isdisjoint(rank_items[1]), (
        f"the ranks overlap on {sorted(set(rank_items[0]) & set(rank_items[1]))}"
    )
    # Disjointness alone would not catch a worker-blind split: with one index per worker
    # process the ranks still stay disjoint, while each rank replays its own items once per
    # worker. The sorted equality is the assertion that fails on a duplicate; the lengths
    # fail when a rank was handed twice its share.
    assert sorted(all_items) == list(range(stream_length)), (
        f"the union is not the stream once each: {sorted(all_items)}"
    )
    assert len(all_items) == stream_length
    assert len(rank_items[0]) == stream_length // world_size
    assert len(rank_items[1]) == stream_length // world_size
