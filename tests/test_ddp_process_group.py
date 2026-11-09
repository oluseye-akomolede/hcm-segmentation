"""The process group comes up on the torch the image actually pins.

A distributed start died with `TypeError: init_process_group() got an unexpected
keyword argument 'device_id'`: the trainer passed `device_id` unconditionally
while torch 2.1's `init_process_group` has no such parameter.  The existing DDP
tests fake `init_process_group`, so nothing here could catch it.  The first test
below therefore builds the keywords with the trainer's own builder and calls the
**real** function, which is the only thing that notices.  See
openspec/changes/ddp-init-process-group-guard/.
"""

import inspect
import socket

import pytest
import torch
import torch.distributed as dist

from ecs_hcm_model.core import trainer

def _free_port():
    """Helper to get an unused port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

@pytest.mark.skipif(
    'device_id' in inspect.signature(dist.init_process_group).parameters,
    reason='this torch accepts device_id: gloo refuses an accelerator-less device_id, so the incident cannot be reproduced here (the target is CI\'s torch 2.1)'
)
def test_a_real_process_group_comes_up_from_the_nccl_kwargs():
    """
    This test reproduces the TypeError from incident where a stray device_id
    reaches Python's argument binding on a torch without the keyword.
    The kwargs come from the trainer's own builder so a mock cannot hide it.
    """
    kwargs = trainer.process_group_kwargs('nccl', torch.device('cpu'))
    kwargs['backend'] = 'gloo'
    kwargs['init_method'] = f'tcp://127.0.0.1:{_free_port()}'
    kwargs['world_size'] = 1
    kwargs['rank'] = 0

    try:
        dist.init_process_group(**kwargs)
        assert dist.get_world_size() == 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def test_the_nccl_kwargs_follow_the_installed_signature():
    """Ensure the kwargs builder respects the torch signature."""
    kwargs = trainer.process_group_kwargs('nccl', torch.device('cpu'))
    assert kwargs['backend'] == 'nccl'
    assert ('device_id' in kwargs) == ('device_id' in inspect.signature(dist.init_process_group).parameters)

def test_the_guard_refuses_the_keyword_when_the_installed_torch_lacks_it(monkeypatch):
    """
    Mock torch 2.1 behavior where init_process_group does not accept device_id.
    The builder must not include device_id in kwargs.
    """
    def torch_21_shaped(backend=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None, group_name='', pg_options=None):
        raise AssertionError('the builder must not call it')
    monkeypatch.setattr(trainer.dist, 'init_process_group', torch_21_shaped)
    assert trainer.process_group_kwargs('nccl', torch.device('cpu')) == {'backend': 'nccl'}

def test_the_guard_passes_the_device_when_the_installed_torch_takes_it(monkeypatch):
    """
    Mock torch 2.3+ behavior where init_process_group accepts device_id.
    The builder must include device_id in kwargs.
    """
    def torch_23_shaped(backend=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None, group_name='', pg_options=None, device_id=None):
        pass  # Mock implementation that accepts device_id
    monkeypatch.setattr(trainer.dist, 'init_process_group', torch_23_shaped)
    device = torch.device('cpu')
    kwargs = trainer.process_group_kwargs('nccl', device)
    assert kwargs['device_id'] is device
    assert kwargs['backend'] == 'nccl'

def test_a_non_nccl_backend_never_carries_a_device_id():
    """Ensure non-nccl backends never carry device_id."""
    assert trainer.process_group_kwargs('gloo', torch.device('cpu')) == {'backend': 'gloo'}

def test_the_rest_of_the_distributed_surface_matches_this_torch():
    """
    Sweep for any other torch > 2.1 API in the distributed path.
    Ensure that all the functions we use are compatible with this torch version.
    """
    sig = inspect.signature(dist.init_process_group)
    assert 'backend' in sig.parameters

    sig = inspect.signature(dist.broadcast_object_list)
    assert {'object_list','src','device'} <= set(sig.parameters)

    sig = inspect.signature(dist.all_reduce)
    assert {'tensor','op'} <= set(sig.parameters)

    sig = inspect.signature(dist.destroy_process_group)
    assert set(sig.parameters) <= {'group'}

    sig = inspect.signature(torch.nn.parallel.DistributedDataParallel.__init__)
    assert {'device_ids','output_device','find_unused_parameters'} <= set(sig.parameters)

    sig = inspect.signature(torch.utils.data.DistributedSampler.__init__)
    assert {'num_replicas','rank','shuffle','seed','drop_last'} <= set(sig.parameters)

    assert hasattr(torch.utils.data.DistributedSampler, 'set_epoch')
    assert hasattr(torch.nn.parallel.DistributedDataParallel, 'no_sync')
