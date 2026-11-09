from types import SimpleNamespace
from ecs_hcm_model.core.GeneratorDataset import GeneratorDataset
import ecs_hcm_model.core.GeneratorDataset as gd_module

def _dataset(monkeypatch, stream_length, rank, world_size, worker_id, num_workers, transform=None):
    monkeypatch.setenv("MINIO_ACCESS_KEY", "dummy")
    monkeypatch.setenv("MINIO_SECRET_KEY", "dummy")
    ds = GeneratorDataset(rank=rank, world_size=world_size)
    ds.arrow_file_generator = lambda shards, s3: iter(range(stream_length))
    ds.transform = transform
    if worker_id is None:
        monkeypatch.setattr(gd_module, "get_worker_info", lambda: None)
    else:
        monkeypatch.setattr(gd_module, "get_worker_info", lambda: SimpleNamespace(id=worker_id, num_workers=num_workers))
    return ds

def test_a_ranks_stream_is_split_across_its_workers(monkeypatch):
    stream_length = 100
    world_size = 1
    rank = 0
    num_workers = 4

    all_items = []
    for worker_id in range(4):
        items = list(_dataset(monkeypatch, stream_length, rank, world_size, worker_id, num_workers))
        expected = list(range(worker_id, 100, 4))
        assert items == expected, f"Worker {worker_id} got {items}, expected {expected}"
        all_items.extend(items)

    assert len(all_items) == 100
    assert sorted(all_items) == list(range(100))

def test_the_rank_split_is_not_a_worker_id(monkeypatch):
    stream_length = 100
    world_size = 2
    num_workers = 2

    for rank, worker_id in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        items = list(_dataset(monkeypatch, stream_length, rank, world_size, worker_id, num_workers))
        expected = list(range(rank * 2 + worker_id, 100, 4))
        assert items == expected, f"Rank {rank}, worker {worker_id} got {items}, expected {expected}"

    # Check that all items are covered exactly once
    all_items = []
    for rank, worker_id in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        all_items.extend(list(_dataset(monkeypatch, stream_length, rank, world_size, worker_id, num_workers)))

    assert len(all_items) == 100
    assert sorted(all_items) == list(range(100))

def test_outside_a_worker_the_split_is_across_ranks_only(monkeypatch):
    stream_length = 100
    world_size = 2
    num_workers = 1
    worker_id = None

    items_0 = list(_dataset(monkeypatch, stream_length, 0, world_size, worker_id, num_workers))
    expected_0 = list(range(0, 100, 2))
    assert items_0 == expected_0

    items_1 = list(_dataset(monkeypatch, stream_length, 1, world_size, worker_id, num_workers))
    expected_1 = list(range(1, 100, 2))
    assert items_1 == expected_1

def test_the_transform_branch_keeps_the_same_split(monkeypatch):
    stream_length = 100
    world_size = 1
    num_workers = 4
    transform = lambda item: item + 1000

    for worker_id in range(4):
        items = list(_dataset(monkeypatch, stream_length, 0, world_size, worker_id, num_workers, transform=transform))
        expected = [item + 1000 for item in range(worker_id, 100, 4)]
        assert items == expected
