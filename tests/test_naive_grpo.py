# -*- coding: utf-8 -*-
"""
Unit test for Naive GRPO advantage computation.
This test verifies that episode-level advantages are correctly computed and broadcast.
"""

import numpy as np
import torch
from verl.trainer.ppo.core_algos import compute_naive_grpo_outcome_advantage


def test_naive_grpo_basic():
    """Test basic functionality of Naive GRPO."""
    batch_size = 4
    response_length = 10
    
    # Create mock data
    episode_rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.int64)
    # Set last token mask to 0 for some samples to simulate padding
    response_mask[0, 8:] = 0
    response_mask[1, 9:] = 0
    
    # Group indices - samples 0,1 in group A, samples 2,3 in group B
    index = np.array(['A', 'A', 'B', 'B'], dtype=object)
    traj_index = np.array(['traj_0', 'traj_1', 'traj_2', 'traj_3'], dtype=object)
    
    # Compute advantages
    advantages, returns = compute_naive_grpo_outcome_advantage(
        episode_rewards=episode_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        norm_adv_by_std_in_grpo=True,
    )
    
    # Check output shapes
    assert advantages.shape == (batch_size, response_length)
    assert returns.shape == (batch_size, response_length)
    
    # For group A: rewards [1.0, 2.0], mean=1.5, std=0.5
    # For group B: rewards [3.0, 4.0], mean=3.5, std=0.5
    
    # Check that advantages are normalized correctly
    # Sample 0: (1.0 - 1.5) / 0.5 = -1.0
    # Sample 1: (2.0 - 1.5) / 0.5 = 1.0
    assert torch.allclose(advantages[0, 0], torch.tensor(-1.0), atol=1e-5)
    assert torch.allclose(advantages[1, 0], torch.tensor(1.0), atol=1e-5)
    
    # Check that advantages are broadcast to all valid tokens
    for i in range(batch_size):
        valid_tokens = response_mask[i] > 0
        if valid_tokens.sum() > 0:
            # All valid tokens should have the same advantage value
            adv_values = advantages[i, valid_tokens]
            assert torch.allclose(adv_values, adv_values[0].expand_as(adv_values), atol=1e-5)
    
    # Check that masked tokens have zero advantage
    assert torch.allclose(advantages[0, 8:], torch.zeros(2), atol=1e-5)
    assert torch.allclose(advantages[1, 9:], torch.zeros(1), atol=1e-5)
    
    print("✓ Basic Naive GRPO test passed")


def test_naive_grpo_without_normalization():
    """Test Naive GRPO without std normalization."""
    batch_size = 4
    response_length = 5
    
    episode_rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.int64)
    index = np.array(['A', 'A', 'B', 'B'], dtype=object)
    traj_index = np.array(['traj_0', 'traj_1', 'traj_2', 'traj_3'], dtype=object)
    
    advantages, returns = compute_naive_grpo_outcome_advantage(
        episode_rewards=episode_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        norm_adv_by_std_in_grpo=False,  # Disable std normalization
    )
    
    # Without std normalization:
    # Sample 0: 1.0 - 1.5 = -0.5
    # Sample 1: 2.0 - 1.5 = 0.5
    assert torch.allclose(advantages[0, 0], torch.tensor(-0.5), atol=1e-5)
    assert torch.allclose(advantages[1, 0], torch.tensor(0.5), atol=1e-5)
    
    print("✓ Naive GRPO without normalization test passed")


def test_naive_grpo_single_sample_group():
    """Test behavior when a group has only one sample."""
    batch_size = 2
    response_length = 5
    
    episode_rewards = np.array([1.0, 2.0], dtype=np.float32)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.int64)
    # Each sample in its own group
    index = np.array(['A', 'B'], dtype=object)
    traj_index = np.array(['traj_0', 'traj_1'], dtype=object)
    
    advantages, returns = compute_naive_grpo_outcome_advantage(
        episode_rewards=episode_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        norm_adv_by_std_in_grpo=True,
    )
    
    # Single sample groups should have advantage = 0
    assert torch.allclose(advantages[0, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(advantages[1, 0], torch.tensor(0.0), atol=1e-5)
    
    print("✓ Naive GRPO single sample group test passed")


if __name__ == "__main__":
    test_naive_grpo_basic()
    test_naive_grpo_without_normalization()
    test_naive_grpo_single_sample_group()
    print("\n✓ All Naive GRPO tests passed!")

