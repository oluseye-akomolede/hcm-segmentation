"""Two-GPU DDP training: the execution model, the reductions, and the artifacts.

The distributed tests run two REAL processes in a REAL gloo process group on CPU
(``tests/ddp_worker.py``), because the properties under test are properties of the
group: which parameters the reducer holds, whether an unfrozen stage is really
synchronised, whether an uneven epoch ends instead of hanging, and what becomes of
the peer when one rank dies. A mock of DDP would only test the mock.

Rows: `TICKET`. Item numbers below are its numbering.
"""

import inspect
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_MODULE = "tests.ddp_worker"

_HAVE_GLOO = torch.distributed.is_available() and torch.distributed.is_gloo_available()
requires_gloo = pytest.mark.skipif(
    not _HAVE_GLOO, reason="torch was built without gloo, so no CPU process group"
)


# --------------------------------------------------------------------------
# The two-process harness
# --------------------------------------------------------------------------


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Run:
    """One scenario, as seen from outside: two exit codes and two result files."""

    def __init__(self, scenario, out_dir, codes, logs, timed_out):
        self.scenario = scenario
        self.out_dir = out_dir
        self.codes = codes
        self.logs = logs
        self.timed_out = timed_out
        self.results = {}
        for rank in (0, 1):
            path = Path(out_dir) / f"rank-{rank}.json"
            self.results[rank] = json.loads(path.read_text()) if path.exists() else None

    def data(self, rank):
        return self.results[rank]["data"]

    def why(self, rank):
        result = self.results[rank]
        if result is None:
            return f"rank {rank} wrote no result (exit {self.codes.get(rank)})"
        return (
            f"rank {rank}: {result.get('error')}\n{result.get('traceback', '')}"
            f"\n--- rank {rank} output ---\n{self.logs[rank]}"
        )


_RUNS = {}


def _run_scenario(scenario, timeout=240):
    """Run both ranks once per pytest session; the results are pure functions of
    the scenario, and a second run would only cost two more processes."""
    if scenario not in _RUNS:
        _RUNS[scenario] = _spawn(scenario, timeout=timeout)
    return _RUNS[scenario]


@pytest.fixture(scope="module", autouse=True)
def _cleanup_scenarios():
    yield
    for run in _RUNS.values():
        shutil.rmtree(run.out_dir, ignore_errors=True)
    _RUNS.clear()


def _spawn(scenario, timeout):
    out_dir = Path(tempfile.mkdtemp(prefix=f"ddp-{scenario}-"))
    base_env = dict(os.environ)
    base_env.update(
        {
            # What torchrun would export to each process.
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_free_port()),
            "WORLD_SIZE": "2",
            "GLOO_SOCKET_IFNAME": "lo",
            # The VM's culture rule: a capped process must not start one thread
            # per host core. Two ranks on one CPU share the budget.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            # Hide any GPU so the backend and the device are the same on every
            # box: this is a CPU/gloo suite (CPU tensors under nccl would fail
            # for reasons that have nothing to do with the code under test).
            "CUDA_VISIBLE_DEVICES": "",
            "ECS_HCM_DDP_OUT": str(out_dir),
            "ECS_HCM_FAILED_RANK_FILE": str(out_dir / "failed_rank"),
            "PYTHONPATH": os.pathsep.join(
                p
                for p in (
                    str(REPO_ROOT / "tests"),
                    str(REPO_ROOT),
                    base_env.get("PYTHONPATH", ""),
                )
                if p
            ),
        }
    )
    procs, handles = {}, {}
    for rank in (0, 1):
        handle = open(out_dir / f"log-rank{rank}.txt", "w", encoding="utf-8")
        handles[rank] = handle
        procs[rank] = subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE, scenario, str(out_dir)],
            cwd=str(REPO_ROOT),
            env=dict(base_env, RANK=str(rank), LOCAL_RANK=str(rank)),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

    codes, timed_out = {}, False
    deadline = time.monotonic() + timeout
    try:
        for rank, proc in procs.items():
            codes[rank] = proc.wait(timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
        for proc in procs.values():
            proc.wait()
    for handle in handles.values():
        handle.close()

    logs = {
        rank: (out_dir / f"log-rank{rank}.txt").read_text(encoding="utf-8", errors="replace")[-4000:]
        for rank in (0, 1)
    }
    return _Run(scenario, out_dir, codes, logs, timed_out)


def _completed(run):
    """Both ranks finished cleanly, or the test fails with their output."""
    if run.timed_out:
        pytest.fail(
            f"{run.scenario}: the two ranks did not finish within the timeout "
            f"(exit codes {run.codes}). Output:\n"
            f"--- rank 0 ---\n{run.logs[0]}\n--- rank 1 ---\n{run.logs[1]}"
        )
    for rank in (0, 1):
        result = run.results[rank]
        assert result is not None, f"{run.scenario}: {run.why(rank)}"
        assert result["ok"], f"{run.scenario}: {run.why(rank)}"
    assert run.results[0]["world_size"] == 2 and run.results[1]["world_size"] == 2, (
        f"{run.scenario}: the ranks did not see a world size of 2"
    )
    return run


# --------------------------------------------------------------------------
# Item 7 -- one wrap, a synchronised unfrozen stage, and the trailing group
# --------------------------------------------------------------------------


@requires_gloo
class TestTwoRankExecution:
    def test_one_wrap_and_the_unfrozen_stage_is_synchronised(self):
        run = _completed(_run_scenario("staged_unfreeze"))
        rank0, rank1 = run.data(0), run.data(1)

        # The two ranks really do read different examples -- otherwise every
        # rank-to-rank comparison below could agree for the wrong reason.
        assert rank0["checksum"] != rank1["checksum"]

        # One wrap for the whole run, taken before the warmup freeze. A wrap
        # after each unfreeze would show two, and a wrapper built after the
        # freeze would exclude the stage that is unfrozen later.
        assert rank0["wraps"] == 1 and rank1["wraps"] == 1, (
            "the model must be wrapped once, before the warmup freeze"
        )

        # The whole-backbone freeze of epoch 0 did not stop the run (that is
        # find_unused_parameters, and `wrap_without_find_unused` pins it), and
        # epoch 1's unfrozen stage really received gradients on both ranks.
        assert rank0["layer4_grads"] > 0 and rank1["layer4_grads"] > 0, (
            "the story being pinned is that the unfrozen stage gets a gradient "
            "on BOTH ranks -- that is what the wrap-before-freeze buys"
        )
        grad0 = torch.tensor(rank0["layer4_grad"])
        grad1 = torch.tensor(rank1["layer4_grad"])
        torch.testing.assert_close(
            grad0,
            grad1,
            rtol=0,
            atol=0,
            msg="the unfrozen stage's gradient must be all-reduced, not local",
        )
        assert rank0["layer4_moved"] and rank1["layer4_moved"]

        # The frozen stage stayed frozen.
        assert rank0["stem_frozen"] and rank1["stem_frozen"]

        # Item 9, in passing: the captured state dict has no wrapper prefix.
        assert not any(k.startswith("module.") for k in rank0["params"]), (
            f"the saved state dict carries DDP's prefix: {sorted(rank0['params'])[:4]}"
        )

        # Item 8, the buffer half. The scenario sets a different running mean
        # on each rank *after* the wrap, so two ranks can only come out of the
        # run holding the same statistics if DDP keeps laying down the owning
        # rank's buffers on every forward: without that sync rank 1 would still
        # be carrying its own 2.0, and the two ranks would report different
        # numbers here.
        assert "backbone.bn.running_mean" in rank0["params"], (
            "the scenario model must actually carry running statistics"
        )
        mean0, mean1 = rank0["bn_running_mean"], rank1["bn_running_mean"]
        tracked0, tracked1 = rank0["bn_tracked"], rank1["bn_tracked"]
        assert mean0 == mean1, (
            f"the ranks' running statistics drifted apart: {mean0} vs {mean1}"
        )
        assert tracked0 == tracked1 > 0, (
            f"the ranks counted different batch totals: {tracked0} vs {tracked1}"
        )

    def test_the_warmup_epoch_needs_find_unused_parameters(self):
        # The justification for find_unused_parameters=True, from the other
        # side: a wrapper built without it cannot run the frozen warmup epoch.
        run = _run_scenario("wrap_without_find_unused")
        assert not run.timed_out, (
            "a rank that under-reduces must fail loudly rather than leave its "
            f"peer waiting:\n{run.logs[0]}\n{run.logs[1]}"
        )
        errors = [
            "\n".join(run.results[rank].get("data", {}).get("raised") or [])
            for rank in (0, 1)
        ]
        assert all(errors), (
            f"both ranks must record the failure, got {errors}\n"
            f"{run.logs[0]}\n{run.logs[1]}"
        )
        assert any("Expected to have finished reduction" in (e or "") for e in errors), (
            f"the failure should be DDP's unused-parameter error, got {errors}"
        )

    def test_a_group_of_micro_batches_syncs_once_at_its_end(self):
        run = _completed(_run_scenario("accum_sync"))
        rank0, rank1 = run.data(0), run.data(1)

        expected = [False, True, False, True, False]
        for rank, data in ((0, rank0), (1, rank1)):
            assert data["flags"] == expected, (
                f"rank {rank}: each micro-batch must sync only at its group's "
                f"end (no_sync on the rest); got {data['flags']}"
            )
            assert data["steps"] == 2, (
                f"rank {rank}: five micro-batches at a group of two step twice"
            )
            # Item 6: the trailing partial group is dropped, not stepped on a
            # rank-local basis, and the drop is observable.
            assert data["discarded"] == 1.0, (
                f"rank {rank}: the leftover micro-batch must be discarded and "
                f"logged, got {data['discarded']}"
            )


# --------------------------------------------------------------------------
# The wrapper is not the model: what the loop reads off it after the wrap
# --------------------------------------------------------------------------


@requires_gloo
class TestAttributeReadsAfterTheWrap:
    def test_test_mode_diagnostics_read_the_module_not_the_wrapper(self):
        # The incident (dags-lead, two-rank speed test): both ranks died ~20s in
        # with `AttributeError: 'DistributedDataParallel' object has no attribute
        # 'segmentation_head'` -- the smoke-mode print read the wrapper, and only
        # test_mode reaches that line, which is why the one-GPU run passed.
        #
        # The same class of read is worse where it is silent: the activation
        # maxima go through a getattr default, so a fold through the wrapper
        # finds nothing and the run logs no maxima rather than failing.
        run = _completed(_run_scenario("test_mode_after_wrap"))
        for rank in (0, 1):
            data = run.data(rank)
            head = data["head_param_tensors"]
            assert data["seg_params_line"] == f"seg params in optimizer: {head} / {head}", (
                f"rank {rank}: the smoke-mode diagnostics must be read from the "
                f"module the optimizer was built from, so every head parameter "
                f"is in a group; got {data['seg_params_line']!r}"
            )
            # Past the test_mode block, and the epoch really trained.
            assert data["steps"] == 2, (
                f"rank {rank}: both batches must reach an optimizer step, got "
                f"{data['steps']}"
            )
            # Those three values were seeded on the module, so they can only get
            # into the epoch's metrics by being read from the module.
            assert data["activation_absmax"] == {
                "train_isc_absmax": 3.5,
                "train_csc_absmax": 1.25,
                "train_isc3_absmax": 0.5,
            }, (
                f"rank {rank}: the epoch's activation maxima must be folded from "
                f"the model's own counters, got {data['activation_absmax']}"
            )


# --------------------------------------------------------------------------
# The amodal outline head under two ranks
# --------------------------------------------------------------------------


@requires_gloo
class TestTheOutlineHeadUnderDDP:
    def test_an_unlabelled_batch_does_not_leave_the_head_outside_the_graph(self):
        # The outline term is a masked mean over the labelled samples only. On a
        # batch where none is labelled it is exactly zero -- and a zero the
        # graph never touches leaves the head's parameters unreduced, which DDP
        # refuses at backward with "Expected to have finished reduction". That
        # failure only exists with more than one rank, so it is tested here.
        run = _completed(_run_scenario("outline_unlabelled_batch"))
        rank0, rank1 = run.data(0), run.data(1)

        for rank, data in ((0, rank0), (1, rank1)):
            # The epoch ran to its end with an unlabelled batch in it
            # (last_epoch is zero-based: one epoch ran).
            assert data["last_epoch"] == 0, f"rank {rank}: the epoch did not finish"
            # Both ranks' labelled batches count: the epoch's counters are
            # summed across the ranks, so 3 images each is 6 in the pooled
            # number -- the unlabelled batch contributes nothing.
            assert data["labelled"] == [6.0], (
                f"rank {rank}: the labelled images must be counted, "
                f"got {data['labelled']}"
            )
            assert len(data["losses"]) == 1, (
                f"rank {rank}: the outline term must be logged every epoch, got "
                f"{data['losses']}"
            )
            assert data["val_full_iou"] is not None, (
                f"rank {rank}: the outline validation numbers must be logged"
            )
        # Pooled epoch values: the same number on both ranks, from different
        # shards. The head ends on the same weights on both ranks for the same
        # reason the loss does -- an average, not two local steps.
        assert rank0["train_loss"] == rank1["train_loss"], (
            "the epoch's mean loss must be reduced across the ranks"
        )
        assert rank0["outline_weight"] == rank1["outline_weight"], (
            "the head must be synchronised across the ranks"
        )


# --------------------------------------------------------------------------
# Item 4 -- what the ranks agree on before anything decides
# --------------------------------------------------------------------------


@requires_gloo
class TestReductions:
    def test_pooled_mean_not_a_mean_of_means(self):
        run = _completed(_run_scenario("reduce_arithmetic"))
        rank0, rank1 = run.data(0), run.data(1)

        # The two ranks contribute counts of 1 and 3, losses summing to 2.0 and
        # 12.0, and 0 and 1 failed images. Both ranks must read the same sums.
        for rank, data in ((0, rank0), (1, rank1)):
            reduced = data["reduced"]
            assert reduced["batch_count"] == 4, f"rank {rank}: {reduced}"
            assert reduced["attempted"] == 4
            assert reduced["oom_skipped"] == 0
            assert reduced["running_loss"] == pytest.approx(14.0)
            assert reduced["running_acc1"] == pytest.approx(2.0)
            assert reduced["running_acc5"] == pytest.approx(4.0)
            # An integer comes back an integer, so the zero-batch check can
            # still compare a summed count against 0.
            assert reduced["train_failed_images"] == 1
            for key in ("batch_count", "attempted", "oom_skipped", "train_failed_images"):
                assert isinstance(reduced[key], int), (
                    f"rank {rank}: {key} came back as {type(reduced[key]).__name__}"
                )

            # 14.0 over 4 batches, which is what one process seeing all four
            # batches would compute. A mean of the ranks' means gives 3.0.
            assert data["pooled_mean"] == pytest.approx(3.5)
            assert data["mean_of_means"] == pytest.approx(3.0), "the wrong answer, for contrast"
            assert data["pooled_mean"] == pytest.approx(sum([2.0, 4.0, 4.0, 4.0]) / 4), (
                "the reduced metric must equal the single-process value"
            )

            # The run's worst unbroken OOM stretch is a max (rank 0 saw 5, rank
            # 1 saw 21), and the preflight's agreement is a min (12.0 against
            # 4.5): a sum would let one rank's health excuse the other's.
            assert data["consecutive_oom"] == 21, "a max, not a sum"
            assert data["fewest_free_gib"] == pytest.approx(4.5), "a min, not a sum"

    def test_the_fraction_gate_is_judged_on_the_run_not_on_a_rank(self):
        """The end-of-epoch fraction check reads the run's counts, not a rank's.

        Each rank attempts 104 batches and skips 2, so no rank can trip
        anything by itself: 104 is under the floor that would let its own
        fraction be judged at all. The run attempted 208 and skipped 4 -- past
        the floor and over the 1% limit -- so the gate fails both ranks. Handed
        one rank's counters instead, the gate stays silent and this test fails,
        which is the discrimination it exists for.

        Every number here is measured from the run -- the batches each rank's
        loader actually handed over and the OOM calls that actually fired -- so
        a drift between the scenario's plan and what ran fails here rather than
        quietly re-checking constants.
        """
        run = _completed(_run_scenario("oom_fraction_gate"))
        data = {rank: run.data(rank) for rank in (0, 1)}

        for rank in (0, 1):
            d = data[rank]
            batches, skips = d["per_rank_batches"], d["per_rank_skips"]
            floor, limit = d["min_batches_for_fraction"], d["max_skip_fraction"]

            # Verify the actual OOM firing matches the planned firing
            assert d["oom_fired_at"] == d["oom_plan"], (
                f"rank {rank}: expected OOM calls {d['oom_plan']}, but actually fired {d['oom_fired_at']}"
            )

            # The run has to have taken some batches and skipped fewer than it took
            assert 0 < skips < batches, (
                f"rank {rank}: must have taken batches and skipped fewer than taken, "
                f"got {batches} batches and {skips} skips"
            )

            # The scenario's premise, asserted so that changing either the
            # numbers or the limits cannot quietly turn this into a test of
            # nothing: one rank's attempts stay under the floor, the run's pass
            # it, and only the run's fraction is over the limit.
            assert batches < floor <= 2 * batches, (
                f"one rank must stay under the floor of {floor}, the run must not: "
                f"{batches} attempted per rank"
            )
            assert (2 * skips) / (2 * batches) > limit, (
                "the run's own fraction must be the one over the limit"
            )

        # Both ranks must measure the same values so that summing is meaningful
        assert data[0]["per_rank_batches"] == data[1]["per_rank_batches"], (
            "both ranks must measure the same number of batches per rank"
        )
        assert data[0]["per_rank_skips"] == data[1]["per_rank_skips"], (
            "both ranks must measure the same number of skips per rank"
        )

        # Compute run-level totals
        run_batches = sum(d["per_rank_batches"] for d in data.values())
        run_skips = sum(d["per_rank_skips"] for d in data.values())

        # The gate must judge the run's summed counts, not individual ranks'
        for rank in (0, 1):
            d = data[rank]
            assert d["raised"], (
                f"rank {rank} finished the epoch: the gate did not judge the "
                f"run's summed counts. {run.why(rank)}"
            )
            assert f"{run_skips}/{run_batches}" in d["error"], (
                f"rank {rank} failed on something other than the run's numbers: "
                f"{d['error']}"
            )
            assert "out of GPU memory" in d["error"]

    def test_validation_runs_once_and_is_handed_to_every_rank(self):
        run = _completed(_run_scenario("validation_once"))
        rank0, rank1 = run.data(0), run.data(1)

        assert rank0["evaluate_calls"] == [0], (
            "validation must be measured once, on the rank that owns the run"
        )
        assert rank1["evaluate_calls"] == [], (
            f"rank 1 must not measure its own validation, got {rank1['evaluate_calls']}"
        )
        # The stub is rank-specific (1.0 on rank 0, 2.0 on rank 1). Both ranks
        # must end up with rank 0's value -- not a second measurement and not
        # the mean of the two, which would be 1.5.
        assert rank0["best_metrics"]["val_loss"] == pytest.approx(1.0)
        assert rank1["best_metrics"]["val_loss"] == pytest.approx(1.0)
        assert rank1["best_metrics"]["val_loss"] != pytest.approx(1.5), (
            "the per-image val metric must not be averaged a second time"
        )
        assert rank0["best_epoch"] == 0 and rank1["best_epoch"] == 0


# --------------------------------------------------------------------------
# Item 5 -- failure and uneven input never hang
# --------------------------------------------------------------------------


@requires_gloo
class TestFailureAndUnevenEpochs:
    def test_an_uneven_epoch_completes_and_pools_its_loss(self):
        run = _completed(_run_scenario("uneven_epoch"))
        rank0, rank1 = run.data(0), run.data(1)

        assert (rank0["n_batches"], rank1["n_batches"]) == (4, 3), (
            "this test is only meaningful if the ranks' epochs really differ"
        )
        assert rank0["last_epoch"] == 0 and rank1["last_epoch"] == 0

        # One epoch, one pooled loss. Without the reduction each rank would log
        # its own mean over its own three or four batches, and two different
        # numbers over different data are equal only by accident.
        assert len(rank0["train_loss"]) == 1 and len(rank1["train_loss"]) == 1
        assert rank0["train_loss"] == pytest.approx(rank1["train_loss"]), (
            f"rank 0 logged {rank0['train_loss']}, rank 1 {rank1['train_loss']}"
        )

    def test_the_next_epoch_starts_from_the_same_state(self):
        # An uneven epoch ends with rank 0 having taken a step that rank 1 did
        # not take, so the two ranks leave the epoch holding different weights
        # unless the boundary puts them back together. If it did not, the second
        # epoch would train two different models and only the all-reduced
        # gradients would be shared -- a silent, permanent divergence.
        run = _completed(_run_scenario("uneven_epochs"))
        rank0, rank1 = run.data(0), run.data(1)

        assert rank0["n_batches"] != rank1["n_batches"], (
            "this test is only meaningful if the ranks' epochs really differ"
        )
        assert (rank0["last_epoch"], rank1["last_epoch"]) == (1, 1), "both epochs must run"
        assert len(rank0["train_loss"]) == 2 and len(rank1["train_loss"]) == 2
        assert rank0["train_loss"] == pytest.approx(rank1["train_loss"])
        assert rank0["param_checksum"] == rank1["param_checksum"], (
            "the ranks finished the run holding different parameters: "
            f"rank 0 {rank0['first_param']}, rank 1 {rank1['first_param']}"
        )

    def test_a_rank_that_dies_ends_both_ranks(self):
        run = _run_scenario("rank_failure", timeout=180)

        assert not run.timed_out, (
            "one rank's terminal error must not leave the other waiting in a "
            f"collective:\n{run.logs[0]}\n{run.logs[1]}"
        )
        assert run.results[1]["ok"] is False, "the failing rank exits non-zero"
        assert run.codes[1] != 0, f"rank 1 exited {run.codes.get(1)}, expected non-zero"
        assert run.codes[0] != 0, (
            f"rank 0 exited {run.codes.get(0)}; a peer's failure must end it too"
        )
        assert run.results[0] is not None
        assert run.results[0]["ok"] is False, (
            "rank 0 cannot have completed the epoch its peer died in"
        )
        assert run.results[0]["group_destroyed"], (
            "the process group must be torn down on the way out (item 5)"
        )


# --------------------------------------------------------------------------
# Item 3 -- one writer
# --------------------------------------------------------------------------


@requires_gloo
class TestOneWriter:
    def test_only_the_main_rank_writes_epoch_checkpoints(self):
        run = _completed(_run_scenario("one_writer"))
        rank0, rank1 = run.data(0), run.data(1)

        # JSON has no tuples, so this is the list form of [(0, 0), (0, 1)]:
        # (rank, epoch) per write.
        assert rank0["writes"] == [[0, 0], [0, 1]], (
            f"the main rank writes each epoch exactly once, got {rank0['writes']}"
        )
        assert rank1["writes"] == [], (
            f"a second rank must write nothing, got {rank1['writes']}"
        )


# --------------------------------------------------------------------------
# Single process: the gating, the walker, and the loader wiring
# --------------------------------------------------------------------------


class TestSingleProcessUnchanged:
    def test_no_env_is_the_single_process_path(self, monkeypatch):
        from ecs_hcm_model.core.trainer import distributed_context_from_env

        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            monkeypatch.delenv(key, raising=False)
        ctx = distributed_context_from_env()
        assert (ctx.rank, ctx.local_rank, ctx.world_size) == (0, 0, 1)
        assert ctx.is_main and not ctx.is_distributed

    def test_a_single_process_run_does_not_probe_cuda_for_a_backend(self, monkeypatch):
        # Caught by CI, from tests/test_coco_mix.py: the entry point logs the
        # COCO mix at step 0 before any GPU probe, by design. A backend chosen
        # in `distributed_context_from_env` probed CUDA every run, so a
        # single-process run touched the GPU before its run record was open --
        # and before `set_env_vars` had set CUDA_DEVICE_ORDER. The backend only
        # matters to `init_process_group_for`, which a one-rank run never
        # reaches, so the probe must not happen at world size one -- and must
        # still happen once a group is real.
        from ecs_hcm_model.core import trainer as trainer_module
        from ecs_hcm_model.core.trainer import distributed_context_from_env

        probes = []
        monkeypatch.setattr(
            trainer_module.torch.cuda,
            "is_available",
            lambda *args, **kwargs: probes.append(1) or False,
        )
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            monkeypatch.delenv(key, raising=False)
        assert distributed_context_from_env().backend == "gloo"
        assert probes == [], "a one-rank run probed CUDA before its run record was open"

        monkeypatch.setenv("WORLD_SIZE", "2")
        assert distributed_context_from_env().backend == "gloo"
        assert len(probes) == 1, "a multi-rank run must still choose its backend"

    def test_the_env_is_read_as_torchrun_writes_it(self, monkeypatch):
        from ecs_hcm_model.core.trainer import distributed_context_from_env

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "2")
        ctx = distributed_context_from_env()
        assert (ctx.rank, ctx.local_rank, ctx.world_size) == (1, 1, 2)
        assert not ctx.is_main and ctx.is_distributed

    def test_a_rank_outside_the_world_size_is_refused(self, monkeypatch):
        # A launcher whose rank numbering disagrees with its world size would
        # otherwise train rank 3 of 2, which is not a run at all.
        from ecs_hcm_model.core.trainer import distributed_context_from_env

        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(ValueError, match="outside WORLD_SIZE"):
            distributed_context_from_env()
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)

    def test_the_walker_catches_sync_batch_norm(self):
        # The old condition was `isinstance(m, torch.nn.BatchNorm2d)`, which is
        # False for SyncBatchNorm (they are siblings under _BatchNorm), so a
        # converted model silently kept its statistics updating.
        from ecs_hcm_model.core.trainer import backbone_bn_eval

        backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3),
            torch.nn.SyncBatchNorm(4),
        )
        backbone.train()
        backbone_bn_eval(backbone)
        assert not backbone[1].training, (
            "a SyncBatchNorm layer must be put in eval mode like a BatchNorm2d"
        )

    def test_a_state_dict_saved_by_a_two_rank_run_loads_into_a_plain_model(self):
        # Item 9, from the consumer's side: the artifact a distributed run
        # writes is read by the single-process loader, so it must not carry
        # DDP's `module.` prefix.
        run = _completed(_run_scenario("staged_unfreeze"))
        saved = Path(run.out_dir) / "best_state.pt"
        state = torch.load(saved, map_location="cpu", weights_only=False)

        assert list(state) == [k for k in state if not k.startswith("module.")], (
            f"a wrapped state dict was written: {sorted(state)[:4]}"
        )
        model = _plain_model_class()()
        model.load_state_dict(state, strict=True)

        # The producer's side of the same claim: the saver the loop itself calls
        # must be handed the unwrapped module. The scenario above writes its own
        # state dict, so without this the loop's hand-off is unpinned -- passing
        # the rank's wrapper instead left every test green.
        writer = _completed(_run_scenario("one_writer"))
        assert writer.data(0)["handed_wrapped"] == [False, False], (
            "the epoch saver was handed the rank's wrapper, not the module: "
            f"{writer.data(0)['handed_wrapped']}"
        )

    def test_a_group_normed_backbone_is_left_training(self):
        # The other half of item 8: the walk must not overreach. The model's
        # grouped normalisation carries no running statistics, so there is
        # nothing to pin -- and pinning it would change the forward for no
        # reason. (The ranks' copies cannot diverge either: no buffer, no
        # divergence.)
        from ecs_hcm_model.core.hcm_nn import hcm_nn
        from ecs_hcm_model.core.trainer import backbone_bn_eval

        backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3),
            torch.nn.BatchNorm2d(4),
        )
        hcm_nn.convert_batchnorm_to_groupnorm(backbone)
        assert not list(backbone.buffers()), (
            "the grouped form must hold no running statistics to diverge"
        )
        backbone.train()
        backbone_bn_eval(backbone)
        assert backbone.training, (
            "a normalisation block with nothing to pin must be left as it was"
        )

    def test_the_stream_is_split_after_the_shuffle(self, monkeypatch):
        """The order is the whole point: HF's streaming `.shuffle()` collapses
        `num_shards` to 1, so a `.shard()` applied to the shuffled stream hands
        rank 0 everything and rank 1 nothing. The split has to be the
        `split_dataset_by_node` one, and it has to come after the shuffle."""
        import ecs_hcm_model.core.model_utils as model_utils

        order = []

        class FakeStream:
            """Enough of an IterableDataset for the loader's streaming branch."""

            num_shards = 1

            def map(self, *args, **kwargs):
                order.append("map")
                return self

            def with_format(self, *args, **kwargs):
                return self

            def cast_column(self, *args, **kwargs):
                return self

            def shuffle(self, buffer_size=None, seed=None):
                order.append(("shuffle", seed))
                return self

        def fake_split(dataset, rank, world_size):
            order.append(("split", rank, world_size))
            return dataset

        monkeypatch.setattr(model_utils, "HFIterableDataset", FakeStream)
        monkeypatch.setattr(model_utils, "load_dataset", lambda *a, **kw: FakeStream())
        # The loader logs in to the Hub on its way in -- a network call, which a
        # test must not make (measured: it is what this test hung on).
        monkeypatch.setattr(model_utils, "login", lambda *a, **kw: None)
        monkeypatch.setattr(model_utils, "logout", lambda *a, **kw: None)
        monkeypatch.setattr(model_utils, "_split_stream_by_rank", fake_split)
        monkeypatch.setattr(
            model_utils.torch.utils.data, "DataLoader", lambda **kw: kw
        )

        model_utils.get_training_dataloader(
            dataset_name="imagenet-1k",
            batch_size=2,
            workers=0,
            global_rank=1,
            world_size=2,
        )

        shuffles = [i for i, e in enumerate(order) if isinstance(e, tuple) and e[0] == "shuffle"]
        splits = [i for i, e in enumerate(order) if isinstance(e, tuple) and e[0] == "split"]
        assert shuffles, f"the streaming path must shuffle: {order}"
        assert ("split", 1, 2) in order, f"rank 1 of 2 must be given a shard: {order}"
        assert shuffles[0] < splits[0], (
            f"the split must come after the shuffle, not before: {order}"
        )

    def test_world_size_one_does_not_split_the_stream(self):
        import ecs_hcm_model.core.model_utils as model_utils

        sentinel = object()
        assert model_utils._split_stream_by_rank(sentinel, 0, 1) is sentinel

    def test_the_local_path_refuses_to_run_distributed(self):
        import ecs_hcm_model.core.model_utils as model_utils

        # The reason is coverage, not duplication: 759bbfe made the local
        # GeneratorDataset split per worker, so the refusal no longer cites
        # duplicated examples and would now be stale if it did. It stays because
        # no cross-rank test and no production caller reach this path.
        with pytest.raises(
            ValueError, match=r"not offered across ranks.*world_size=2, rank=1"
        ) as refused:
            model_utils.get_training_dataloader(
                dataset_name="imagenet-1k",
                batch_size=2,
                workers=1,
                is_dataset_local=True,
                global_rank=1,
                world_size=2,
            )

        # The reason is part of the spec's scenario, so pin it and not just the
        # refusal: a message that fell back to citing duplicated examples would
        # pass the match above.
        assert "no cross-rank test coverage and no production caller" in str(refused.value), (
            f"the refusal must give its real reason, got: {refused.value}"
        )


def _plain_model_class():
    """A plain model of the same shape the worker trained, without DDP."""
    from conftest import TinyClassifier

    if str(REPO_ROOT / "tests") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
    from ddp_worker import _TwoStageBackbone

    class PlainStaged(TinyClassifier):
        def __init__(self):
            super().__init__()
            self.backbone = _TwoStageBackbone()

    return PlainStaged


# --------------------------------------------------------------------------
# The entry point: gating, and which rank builds what (items 1 and 3)
# --------------------------------------------------------------------------


class TestEntryPointGating:
    """`train_hcm_imagenet1k` without a real process group.

    The rendezvous is faked (`dist.init_process_group` is replaced, so the real
    helper's decision to call it is what is observed) -- but the rank gating,
    the logger, the upload and the loader arguments are the production ones.
    """

    def _wire(self, monkeypatch):
        from conftest import RecordingLogger, RecordingUploader, TinyClassifier

        built_loggers, loader_calls, val_calls, inits, destructions = [], [], [], [], []
        captured = {}
        logger = RecordingLogger()

        def fake_logger(**kwargs):
            built_loggers.append(kwargs)
            return logger

        def fake_train_loader(**kwargs):
            loader_calls.append(kwargs)
            return [torch.randn(1)]

        def fake_val_loader(**kwargs):
            val_calls.append(kwargs)
            return [torch.randn(1)]

        def fake_train(**kwargs):
            captured.update(kwargs)
            return {
                "best_epoch": 0,
                "best_score": 1.0,
                "best_state": TinyClassifier().state_dict(),
                "best_metrics": {"val_acc1": 1.0},
                "best_train_loss": 0.1,
                "last_epoch": 0,
            }

        monkeypatch.setattr("ecs_hcm_model.core.trainer.logger_config", lambda **kw: {"fake": True})
        monkeypatch.setattr("ecs_hcm_model.core.trainer.MLFlowHCMLogger", fake_logger)
        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.ModelUploaderAndDownloader",
            lambda **kw: RecordingUploader(),
        )
        monkeypatch.setattr("ecs_hcm_model.core.trainer.get_training_dataloader", fake_train_loader)
        monkeypatch.setattr("ecs_hcm_model.core.trainer.get_validation_dataloader", fake_val_loader)
        monkeypatch.setattr("ecs_hcm_model.core.trainer.load_hcm_model", lambda **kw: TinyClassifier())
        monkeypatch.setattr("ecs_hcm_model.core.trainer.set_env_vars", lambda *a, **k: None)
        monkeypatch.setattr("ecs_hcm_model.core.trainer.train_model_for_epochs", fake_train)
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr(
            "torch.distributed.init_process_group",
            lambda **kw: inits.append(kw),
        )
        monkeypatch.setattr(
            "torch.distributed.destroy_process_group",
            lambda: destructions.append(1),
        )
        return built_loggers, loader_calls, val_calls, inits, captured, logger

    def test_a_run_without_the_env_takes_the_single_process_path(self, monkeypatch):
        from ecs_hcm_model.core.trainer import train_hcm_imagenet1k

        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            monkeypatch.delenv(key, raising=False)
        built_loggers, loader_calls, _val, inits, captured, logger = self._wire(monkeypatch)

        train_hcm_imagenet1k(max_epochs=1, batch_size=2, test_mode=True)

        ctx = captured["distributed"]
        assert (ctx.rank, ctx.world_size, ctx.is_main) == (0, 1, True)
        assert inits == [], "a single-process run must not create a process group"
        assert logger.finish_statuses() == ["FINISHED"]
        assert all(call.get("world_size", 1) == 1 for call in loader_calls), (
            "the single-process loader call must keep its default world size"
        )
        assert len(built_loggers) == 1, "one run, one logger"

    def test_a_non_main_rank_builds_no_logger_and_uploads_nothing(self, monkeypatch):
        from ecs_hcm_model.core.trainer import train_hcm_imagenet1k

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "3")
        built_loggers, loader_calls, val_calls, inits, captured, logger = self._wire(monkeypatch)

        train_hcm_imagenet1k(max_epochs=1, batch_size=2, test_mode=True)

        ctx = captured["distributed"]
        assert (ctx.rank, ctx.world_size, ctx.is_main) == (1, 3, False)
        assert len(inits) == 1 and inits[0]["backend"] == "gloo", (
            f"the group must come up exactly once for a multi-rank run: {inits}"
        )
        assert built_loggers == [], (
            "only the rank that owns the run may open the MLflow run: "
            f"{built_loggers}"
        )
        assert logger.finish_statuses() == [], "a second rank must write no run status"
        assert loader_calls, "the training loaders must still be built"
        assert all(
            call.get("world_size") == 3 and call.get("global_rank") == 1
            for call in loader_calls
        ), f"every training loader must be told its rank: {loader_calls}"
        assert val_calls, "the validation loaders must still be built"
        assert all(
            call.get("world_size") == 1 and call.get("global_rank") == 0
            for call in val_calls
        ), (
            "validation stays replicated in this version: every rank's val "
            f"loader covers the whole validation set, got {val_calls}"
        )

    def test_the_failed_run_names_the_peer_that_died(self, monkeypatch, tmp_path):
        # The run says which rank was lost, not which rank noticed. Rank 1 dies
        # leaving its number in the shared marker (the real peer does this in
        # its own error arm); rank 0 owns the run and must read that number
        # before writing its own over it.
        from ecs_hcm_model.core.trainer import record_failed_rank, train_hcm_imagenet1k

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("ECS_HCM_FAILED_RANK_FILE", str(tmp_path / "failed_rank"))
        _built, _loaders, _val, _inits, _captured, logger = self._wire(monkeypatch)

        def failing_train(**kwargs):
            record_failed_rank(1)
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        monkeypatch.setattr(
            "ecs_hcm_model.core.trainer.train_model_for_epochs", failing_train
        )

        # It must fail, and loudly.
        with pytest.raises(RuntimeError, match="out of memory"):
            train_hcm_imagenet1k(max_epochs=1, batch_size=2, test_mode=True)

        assert logger.finish_statuses() == ["FAILED"], (
            "a distributed failure must end the MLflow run FAILED, not abandon it"
        )
        assert logger.failed_rank == 1, (
            "rank 0 owns the run, so the tag must name the rank that actually "
            f"failed; got {logger.failed_rank}"
        )

    def test_the_device_the_group_and_the_preflight_happen_in_that_order(
        self, monkeypatch, tmp_path
    ):
        """The order the CUDA path depends on, observed on a CPU-only box.

        The harness cannot observe device *selection* -- there is no GPU here --
        but it can observe the order of the calls that selection feeds: nccl
        binds the rank's device when the group comes up, and the preflight is
        itself a collective, so it can only run inside the group. It also pins
        that the group is taken down on the error path.
        """
        from ecs_hcm_model.core.trainer import train_hcm_imagenet1k

        class _Stop(RuntimeError):
            """Stands in for the rest of the run once the order is known."""

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "2")
        # The run records its own failure in the shared marker; keep it out of
        # /tmp so a test never leaves state a later run could read.
        monkeypatch.setenv("ECS_HCM_FAILED_RANK_FILE", str(tmp_path / "failed_rank"))
        _b, _l, _v, _inits, _c, logger = self._wire(monkeypatch)

        order = []
        state = {"up": False, "destroyed": 0}

        # The trainer passes `device_id` only when the function it is about to
        # call declares it (torch 2.1 does not; 2.3+ does), so the stand-in here
        # has to declare the parameters the real function declares.  A stand-in
        # that swallows `**kwargs` reads as "this torch has no device_id" to that
        # guard -- which is precisely how the unconditional keyword went
        # unnoticed until it crashed a real distributed start.
        _declares_device_id = "device_id" in inspect.signature(
            torch.distributed.init_process_group
        ).parameters

        def _init(backend, device_id):
            state["up"] = True  # from here, `is_initialized` is true
            order.append(("init", backend, str(device_id)))

        def fake_init(backend=None, init_method=None, timeout=None, world_size=-1,
                      rank=-1, store=None, group_name="", pg_options=None):
            _init(backend, None)

        def fake_init_device_id(backend=None, init_method=None, timeout=None,
                                world_size=-1, rank=-1, store=None, group_name="",
                                pg_options=None, device_id=None):
            _init(backend, device_id)

        def fake_mem_get_info():
            order.append(("preflight",))
            raise _Stop("the preflight is reached")

        def fake_destroy():
            state["destroyed"] += 1

        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            "torch.cuda.set_device", lambda device: order.append(("set_device", str(device)))
        )
        monkeypatch.setattr("torch.cuda.mem_get_info", fake_mem_get_info)
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: state["up"])
        monkeypatch.setattr(
            "torch.distributed.init_process_group",
            fake_init_device_id if _declares_device_id else fake_init,
        )
        monkeypatch.setattr("torch.distributed.destroy_process_group", fake_destroy)

        with pytest.raises(_Stop):
            train_hcm_imagenet1k(max_epochs=1, batch_size=2, test_mode=True)

        assert order == [
            ("set_device", "cuda:1"),
            # nccl binds the rank's device when the group comes up, on a torch
            # whose init takes the keyword; on one whose doesn't, the rank's
            # device is bound by set_device alone and no device_id is passed.
            ("init", "nccl", "cuda:1" if _declares_device_id else "None"),
            ("preflight",),
        ], (
            "the rank's device must be bound before the group comes up (nccl "
            "binds it at init) and the group must exist before the preflight's "
            f"collective; got {order}"
        )
        assert state["destroyed"] == 1, (
            "a run that fails must still take its process group down, or its "
            "peers wait out a collective timeout instead of failing fast"
        )
        # This rank is not the main one, so it must leave the run record alone
        # -- rank 0 owns the MLflow run, and two writers would fight over it --
        # while still leaving its number where rank 0 can read it.
        assert logger.finish_statuses() == [], (
            f"a non-main rank ended the run record: {logger.finish_statuses()}"
        )
        assert logger.failed_rank is None
        marker = (tmp_path / "failed_rank").read_text().strip()
        assert marker == "1", (
            f"the dying rank must leave its number for rank 0 to read; got {marker!r}"
        )
