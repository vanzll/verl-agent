"""
Tests for the 5 GTPO bug fixes in dp_actor.py.

These tests verify the logic of update_policy() without requiring
a real model, FSDP, or distributed setup. We mock the heavy parts
and focus on the data flow correctness.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from verl.workers.actor.dp_actor import (
    DataParallelPPOActor,
    split_micro_batches_by_trajectory,
)
from verl import DataProto


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Build a minimal OmegaConf config that DataParallelPPOActor expects."""
    defaults = dict(
        use_remove_padding=False,
        use_fused_kernels=False,
        ulysses_sequence_parallel_size=1,
        use_torch_compile=False,
        grad_clip=1.0,
        clip_ratio=0.2,
        clip_ratio_low=None,
        clip_ratio_high=None,
        clip_ratio_c=3.0,
        entropy_coeff=0.0,
        loss_agg_mode="mean",
        use_kl_loss=False,
        kl_loss_type="kl",
        kl_loss_coef=0.1,
        ppo_mini_batch_size=8,
        ppo_micro_batch_size_per_gpu=4,
        ppo_max_token_len_per_gpu=512,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        pure_on_policy=True,
        policy_loss={"loss_mode": "gtpo"},
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def _make_dataproto(batch_size=8, response_length=4, traj_uids=None):
    """Build a minimal DataProto with all keys update_policy() expects."""
    tensors = {
        "input_ids": torch.randint(0, 1000, (batch_size, 10 + response_length)),
        "responses": torch.randint(0, 1000, (batch_size, response_length)),
        "attention_mask": torch.ones(batch_size, 10 + response_length, dtype=torch.long),
        "position_ids": torch.arange(10 + response_length).unsqueeze(0).expand(batch_size, -1),
        "old_log_probs": torch.randn(batch_size, response_length),
        "advantages": torch.randn(batch_size, response_length),
    }
    non_tensors = {}
    if traj_uids is not None:
        non_tensors["traj_uid"] = np.array(traj_uids)

    data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    data.meta_info["temperature"] = 1.0
    data.meta_info["micro_batch_size"] = 4
    data.meta_info["use_dynamic_bsz"] = False
    return data


def _read_dp_actor_source():
    """Read dp_actor.py source directly from file (avoids decorator wrapping issues)."""
    import pathlib
    dp_actor_path = pathlib.Path(__file__).resolve().parent.parent / "verl" / "workers" / "actor" / "dp_actor.py"
    return dp_actor_path.read_text()


# ===========================================================================
# Test: split_micro_batches_by_trajectory (unit-level, no mocking needed)
# ===========================================================================

class TestSplitMicroBatchesByTrajectory:
    """Tests for the trajectory-aware micro-batch splitting helper."""

    def test_basic_split_keeps_trajectories_intact(self):
        """Trajectories should never be split across micro-batches."""
        batch_size = 8
        response_length = 4
        traj_uids = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        tensors = {
            "responses": torch.randn(batch_size, response_length),
            "input_ids": torch.randn(batch_size, 10),
        }
        batch = TensorDict(tensors, batch_size=[batch_size])

        # target micro_batch_size=3 -> naive split would cut traj 0 at index 3
        micro_batches = split_micro_batches_by_trajectory(
            batch, target_micro_batch_size=3, traj_uids=traj_uids
        )

        # Should get 2 micro-batches of 4 each (whole trajectories)
        assert len(micro_batches) == 2
        assert micro_batches[0].batch_size[0] == 4
        assert micro_batches[1].batch_size[0] == 4

    def test_no_traj_uids_falls_back_to_standard_split(self):
        """Without traj_uids, should fall back to normal split."""
        batch_size = 8
        tensors = {"responses": torch.randn(batch_size, 4)}
        batch = TensorDict(tensors, batch_size=[batch_size])

        micro_batches = split_micro_batches_by_trajectory(
            batch, target_micro_batch_size=3, traj_uids=None
        )
        # Standard split of 8 into chunks of 3 -> [3, 3, 2]
        assert len(micro_batches) == 3

    def test_single_trajectory(self):
        """Single trajectory should result in a single micro-batch."""
        batch_size = 6
        traj_uids = np.array([0, 0, 0, 0, 0, 0])
        tensors = {"responses": torch.randn(batch_size, 4)}
        batch = TensorDict(tensors, batch_size=[batch_size])

        micro_batches = split_micro_batches_by_trajectory(
            batch, target_micro_batch_size=2, traj_uids=traj_uids
        )
        # All same traj -> must stay together
        assert len(micro_batches) == 1
        assert micro_batches[0].batch_size[0] == 6

    def test_each_sample_own_trajectory(self):
        """Each sample is its own trajectory -> standard splitting."""
        batch_size = 6
        traj_uids = np.array([0, 1, 2, 3, 4, 5])
        tensors = {"responses": torch.randn(batch_size, 4)}
        batch = TensorDict(tensors, batch_size=[batch_size])

        micro_batches = split_micro_batches_by_trajectory(
            batch, target_micro_batch_size=2, traj_uids=traj_uids
        )
        assert len(micro_batches) == 3
        for mb in micro_batches:
            assert mb.batch_size[0] == 2


# ===========================================================================
# Test: Bug 2 -- pure_on_policy assertion
# ===========================================================================

class TestBug2PureOnPolicyAssertion:
    """GTPO must require pure_on_policy=True."""

    def test_gtpo_without_pure_on_policy_raises(self):
        """GTPO + pure_on_policy=False should raise AssertionError."""
        config = _make_config(pure_on_policy=False)
        module = MagicMock()
        optimizer = MagicMock()
        actor = DataParallelPPOActor(config, module, optimizer)

        data = _make_dataproto(batch_size=8, traj_uids=[0, 0, 1, 1, 2, 2, 3, 3])

        with pytest.raises(AssertionError, match="GTPO requires pure_on_policy=True"):
            actor.update_policy(data)

    def test_non_gtpo_without_pure_on_policy_no_error(self):
        """Non-GTPO modes should not require pure_on_policy."""
        config = _make_config(
            pure_on_policy=False,
            policy_loss={"loss_mode": "vanilla"},
        )
        module = MagicMock()
        optimizer = MagicMock()
        actor = DataParallelPPOActor(config, module, optimizer)

        data = _make_dataproto(batch_size=8)
        # Should not raise AssertionError about pure_on_policy
        # Will fail at forward pass (mocked model), that's fine
        try:
            actor.update_policy(data)
        except AssertionError as e:
            if "pure_on_policy" in str(e):
                pytest.fail(f"Should not raise pure_on_policy assertion for vanilla mode: {e}")
        except Exception:
            pass  # Other errors expected with mocked model


# ===========================================================================
# Test: Bug 1 -- traj_uid_for_minibatch assigned before use
# ===========================================================================

class TestBug1TrajUidOrdering:
    """traj_uid_for_minibatch must be populated before micro-batch splitting."""

    def test_traj_uid_passed_to_split_function(self):
        """Verify split_micro_batches_by_trajectory receives valid traj_uids, not None."""
        config = _make_config(
            pure_on_policy=True,
            ppo_micro_batch_size_per_gpu=4,
        )
        module = MagicMock()
        optimizer = MagicMock()
        actor = DataParallelPPOActor(config, module, optimizer)

        traj_uids = [0, 0, 0, 0, 1, 1, 1, 1]
        data = _make_dataproto(batch_size=8, traj_uids=traj_uids)

        captured_args = {}

        original_split = split_micro_batches_by_trajectory

        def spy_split(mini_batch, target_micro_batch_size, traj_uid_key="traj_uid", traj_uids=None):
            captured_args["traj_uids"] = traj_uids
            return original_split(mini_batch, target_micro_batch_size, traj_uid_key, traj_uids)

        with patch("verl.workers.actor.dp_actor.split_micro_batches_by_trajectory", side_effect=spy_split):
            with patch("verl.workers.actor.dp_actor.dist") as mock_dist:
                mock_dist.all_reduce = MagicMock()
                try:
                    actor.update_policy(data)
                except Exception:
                    pass  # Mocked model will fail; we only care about captured_args

        # The key assertion: traj_uids should NOT be None
        assert "traj_uids" in captured_args, "split_micro_batches_by_trajectory was never called"
        assert captured_args["traj_uids"] is not None, (
            "Bug 1 regression: traj_uids was None when passed to split_micro_batches_by_trajectory"
        )
        np.testing.assert_array_equal(captured_args["traj_uids"], traj_uids)


# ===========================================================================
# Test: Bug 3 -- n_micro_batches_in_minibatch always defined (source check)
# ===========================================================================

class TestBug3NMicroBatchesDefined:
    """n_micro_batches_in_minibatch must be defined for all code paths."""

    def test_unified_assignment_exists_before_zero_grad(self):
        """Verify n_micro_batches_in_minibatch = len(micro_batches) is in the source,
        outside the if/elif/else branches, before zero_grad."""
        source = _read_dp_actor_source()

        # The unified assignment should exist at the correct indentation level
        # (same level as the if/elif/else, not inside a branch)
        assert "n_micro_batches_in_minibatch = len(micro_batches)" in source, (
            "Bug 3 regression: n_micro_batches_in_minibatch not unified after branches"
        )

        # Verify ordering: unified assignment comes before actor_optimizer.zero_grad()
        lines = source.split("\n")
        unified_line = None
        zero_grad_line = None
        for i, line in enumerate(lines):
            if "n_micro_batches_in_minibatch = len(micro_batches)" in line:
                unified_line = i
            if "self.actor_optimizer.zero_grad()" in line and unified_line is not None and zero_grad_line is None:
                zero_grad_line = i
                break

        assert unified_line is not None, "Could not find unified assignment"
        assert zero_grad_line is not None, "Could not find zero_grad after unified assignment"
        assert unified_line < zero_grad_line, (
            "n_micro_batches_in_minibatch must be assigned before zero_grad()"
        )


# ===========================================================================
# Test: Bug 4 -- Dummy micro-batch traj_index handling (source check)
# ===========================================================================

class TestBug4DummyTrajIndex:
    """Dummy micro-batches must use placeholder traj_index."""

    def test_is_dummy_check_before_traj_uid_in_gtpo_block(self):
        """In the GTPO traj_index block, is_dummy must be checked first."""
        source = _read_dp_actor_source()

        assert "is_dummy" in source, "Bug 4 regression: is_dummy check missing in source"

        # Find the GTPO traj_index block and verify is_dummy comes first
        lines = source.split("\n")
        in_gtpo_traj_block = False
        found_is_dummy_first = False
        for line in lines:
            stripped = line.strip()
            # Enter the GTPO block where traj_index is built
            if 'elif loss_mode == "gtpo":' in stripped:
                in_gtpo_traj_block = True
                continue
            if in_gtpo_traj_block:
                if stripped.startswith("if is_dummy"):
                    found_is_dummy_first = True
                    break
                if "traj_uid" in stripped and "traj_index" in stripped:
                    # Found traj_uid lookup before is_dummy -> bug
                    break
                if stripped.startswith("else:") or stripped.startswith("elif"):
                    # Exited block without finding either
                    break

        assert found_is_dummy_first, (
            "Bug 4 regression: is_dummy check must come before traj_uid lookup in GTPO traj_index block"
        )

    def test_dummy_uses_np_zeros(self):
        """Dummy path should use np.zeros for placeholder traj_index."""
        source = _read_dp_actor_source()
        # After "if is_dummy:", the next meaningful line should use np.zeros
        lines = source.split("\n")
        found_dummy_zeros = False
        in_dummy_block = False
        for line in lines:
            stripped = line.strip()
            if "if is_dummy:" in stripped:
                in_dummy_block = True
                continue
            if in_dummy_block:
                if "np.zeros" in stripped and "traj_index" in stripped:
                    found_dummy_zeros = True
                    break
                if stripped and not stripped.startswith("#"):
                    break  # first non-comment line after is_dummy

        assert found_dummy_zeros, "Bug 4 regression: dummy path should use np.zeros for traj_index"


# ===========================================================================
# Test: Bug 5 -- metrics dict doesn't shadow `data` (source check)
# ===========================================================================

class TestBug5MetricsDataRename:
    """Metrics dict variable must not shadow the `data` micro-batch variable."""

    def test_no_data_shadowing_after_backward(self):
        """After loss.backward(), metrics should use `metrics_data`, not `data`."""
        source = _read_dp_actor_source()

        lines = source.split("\n")
        after_backward = False
        for line in lines:
            stripped = line.strip()
            if "loss.backward()" in stripped:
                after_backward = True
                continue
            if after_backward:
                if (stripped.startswith("data=") or stripped.startswith("data =")) and "actor/" in stripped:
                    pytest.fail(
                        f"Bug 5 regression: Found 'data = {{...actor/...}}' after loss.backward(). "
                        f"Should be 'metrics_data'. Line: {stripped}"
                    )
                if "append_to_dict(metrics, data)" in stripped:
                    pytest.fail(
                        f"Bug 5 regression: Found 'append_to_dict(metrics, data)' after loss.backward(). "
                        f"Should be 'append_to_dict(metrics, metrics_data)'. Line: {stripped}"
                    )
                if "_optimizer_step" in stripped:
                    after_backward = False

    def test_metrics_data_used_in_append(self):
        """All append_to_dict calls after backward should use metrics_data."""
        source = _read_dp_actor_source()
        assert "append_to_dict(metrics, metrics_data)" in source, (
            "Bug 5 regression: metrics_data not used in append_to_dict"
        )


# ===========================================================================
# Integration test: full update_policy flow with mocked forward
# ===========================================================================

class TestGTPOUpdatePolicyIntegration:
    """End-to-end test of update_policy with GTPO, using mocked forward pass."""

    @patch("verl.workers.actor.dp_actor.dist")
    @patch("verl.workers.actor.dp_actor.get_torch_device")
    def test_gtpo_update_policy_completes(self, mock_get_device, mock_dist):
        """Full GTPO update_policy should complete without errors."""
        batch_size = 8
        response_length = 4

        # Mock device to use CPU
        mock_device = MagicMock()
        mock_device.current_device.return_value = "cpu"
        mock_get_device.return_value = mock_device

        config = _make_config(
            pure_on_policy=True,
            ppo_micro_batch_size_per_gpu=4,
            ppo_epochs=1,
        )

        module = MagicMock()
        optimizer = MagicMock()

        actor = DataParallelPPOActor(config, module, optimizer)

        # Mock _forward_micro_batch to return CPU tensors
        def mock_forward(micro_batch, temperature, calculate_entropy=False):
            bs = micro_batch["responses"].shape[0]
            log_probs = torch.randn(bs, response_length, requires_grad=True)
            entropy = torch.randn(bs, response_length) if calculate_entropy else None
            return entropy, log_probs

        actor._forward_micro_batch = mock_forward
        actor._optimizer_step = MagicMock(return_value=torch.tensor(0.5))

        mock_dist.all_reduce = MagicMock()
        mock_dist.ReduceOp.MAX = 0

        traj_uids = [0, 0, 0, 0, 1, 1, 1, 1]
        data = _make_dataproto(batch_size=batch_size, response_length=response_length, traj_uids=traj_uids)

        metrics = actor.update_policy(data)

        assert isinstance(metrics, dict)
        assert "actor/pg_loss" in metrics, (
            f"Expected actor/pg_loss in metrics, got: {list(metrics.keys())}"
        )
        assert "actor/gradient_accumulation" in metrics

    @patch("verl.workers.actor.dp_actor.dist")
    def test_gtpo_traj_index_correctness(self, mock_dist):
        """Verify traj_index passed to compute_policy_loss_gtpo is correct."""
        batch_size = 8
        response_length = 4
        config = _make_config(
            pure_on_policy=True,
            ppo_micro_batch_size_per_gpu=4,
            ppo_epochs=1,
        )

        module = MagicMock()
        optimizer = MagicMock()
        actor = DataParallelPPOActor(config, module, optimizer)

        def mock_forward(micro_batch, temperature, calculate_entropy=False):
            bs = micro_batch["responses"].shape[0]
            log_probs = torch.randn(bs, response_length, requires_grad=True)
            return None, log_probs

        actor._forward_micro_batch = mock_forward
        actor._optimizer_step = MagicMock(return_value=torch.tensor(0.5))

        mock_dist.all_reduce = MagicMock()
        mock_dist.ReduceOp.MAX = 0

        traj_uids = [0, 0, 0, 0, 1, 1, 1, 1]
        data = _make_dataproto(batch_size=batch_size, response_length=response_length, traj_uids=traj_uids)

        captured_traj_indices = []

        def spy_gtpo_loss(old_log_prob, log_prob, advantages, response_mask,
                          traj_index, cliprange, cliprange_low, cliprange_high, clip_ratio_c):
            captured_traj_indices.append(np.array(traj_index))
            pg_loss = torch.tensor(0.1, requires_grad=True)
            return pg_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        with patch("verl.workers.actor.dp_actor.compute_policy_loss_gtpo", side_effect=spy_gtpo_loss):
            metrics = actor.update_policy(data)

        assert len(captured_traj_indices) > 0, "compute_policy_loss_gtpo was never called"

        # Verify traj_indices are valid
        for ti in captured_traj_indices:
            unique_vals = np.unique(ti)
            assert len(unique_vals) <= 2, (
                f"Micro-batch contains too many trajectories: {unique_vals}"
            )

    @patch("verl.workers.actor.dp_actor.dist")
    def test_vanilla_mode_unaffected(self, mock_dist):
        """Vanilla loss mode should still work correctly after GTPO changes."""
        batch_size = 8
        response_length = 4
        config = _make_config(
            pure_on_policy=False,
            policy_loss={"loss_mode": "vanilla"},
            ppo_micro_batch_size_per_gpu=4,
            ppo_epochs=1,
        )

        module = MagicMock()
        optimizer = MagicMock()
        actor = DataParallelPPOActor(config, module, optimizer)

        def mock_forward(micro_batch, temperature, calculate_entropy=False):
            bs = micro_batch["responses"].shape[0]
            log_probs = torch.randn(bs, response_length, requires_grad=True)
            return None, log_probs

        actor._forward_micro_batch = mock_forward
        actor._optimizer_step = MagicMock(return_value=torch.tensor(0.5))

        mock_dist.all_reduce = MagicMock()
        mock_dist.ReduceOp.MAX = 0

        data = _make_dataproto(batch_size=batch_size, response_length=response_length)

        def spy_vanilla_loss(old_log_prob, log_prob, advantages, response_mask,
                             cliprange, cliprange_low, cliprange_high, clip_ratio_c, loss_agg_mode):
            pg_loss = torch.tensor(0.1, requires_grad=True)
            return pg_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        with patch("verl.workers.actor.dp_actor.compute_policy_loss", side_effect=spy_vanilla_loss):
            metrics = actor.update_policy(data)

        assert isinstance(metrics, dict)
        assert "actor/pg_loss" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
