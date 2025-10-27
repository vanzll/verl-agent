# -*- coding: utf-8 -*-
"""
Unit test for Advanced GRPO advantage computation.
This test verifies that token-level advantages sum to zero within each group.
"""

import numpy as np
import torch
from verl.trainer.ppo.core_algos import compute_advanced_grpo_outcome_advantage


def test_advanced_grpo_zero_sum_property():
    """Test that token-level advantages sum to zero within each group."""
    batch_size = 4
    response_length = 10
    
    # Create mock data with different episode rewards
    episode_rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.int64)
    
    # Group indices - samples 0,1 in group A, samples 2,3 in group B
    index = np.array(['A', 'A', 'B', 'B'], dtype=object)
    traj_index = np.array(['traj_0', 'traj_1', 'traj_2', 'traj_3'], dtype=object)
    
    # Compute advantages
    advantages, returns = compute_advanced_grpo_outcome_advantage(
        episode_rewards=episode_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        norm_adv_by_std_in_grpo=True,
    )
    
    # Check output shapes
    assert advantages.shape == (batch_size, response_length)
    assert returns.shape == (batch_size, response_length)
    
    # KEY TEST: Sum of advantages within each group should be approximately zero
    group_a_indices = [0, 1]
    group_b_indices = [2, 3]
    
    group_a_sum = sum([advantages[i, response_mask[i] > 0].sum().item() for i in group_a_indices])
    group_b_sum = sum([advantages[i, response_mask[i] > 0].sum().item() for i in group_b_indices])
    
    print(f"Group A advantage sum: {group_a_sum}")
    print(f"Group B advantage sum: {group_b_sum}")
    
    # Should be very close to zero (within numerical precision)
    assert abs(group_a_sum) < 1e-4, f"Group A sum should be ~0, got {group_a_sum}"
    assert abs(group_b_sum) < 1e-4, f"Group B sum should be ~0, got {group_b_sum}"
    
    print("✓ Zero-sum property verified for both groups")


def test_advanced_grpo_token_level_normalization():
    """Test that all tokens in the same episode have the same advantage."""
    batch_size = 2
    response_length = 5
    
    # Two episodes with different rewards
    episode_rewards = np.array([1.0, 2.0], dtype=np.float32)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.int64)
    index = np.array(['A', 'A'], dtype=object)
    traj_index = np.array(['traj_0', 'traj_1'], dtype=object)
    
    advantages, returns = compute_advanced_grpo_outcome_advantage(
        episode_rewards=episode_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        norm_adv_by_std_in_grpo=True,
    )
    
    # All tokens in episode 0 should have the same advantage
    ep0_advantages = advantages[0, response_mask[0] > 0]
    assert torch.allclose(ep0_advantages, ep0_advantages[0].expand_as(ep0_advantages), atol=1e-5)
    
    # All tokens in episode 1 should have the same advantage
    ep1_advantages = advantages[1, response_mask[1] > 0]
    assert torch.allclose(ep1_advantages, ep1_advantages[0].expand_as(ep1_advantages), atol=1e-5)
    
    # But episodes should have different advantages
    assert not torch.allclose(advantages[0, 0], advantages[1, 0], atol=1e-5)
    
    print("✓ Token-level consistency verified")


def test_advanced_grpo_with_varying_lengths():
    """Test Advanced GRPO with varying response lengths."""
    batch_size = 4
    response_length = 10
    
    episode_rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.int64)
    
    # Set different valid lengths
    response_mask[0, 5:] = 0  # 5 valid tokens
    response_mask[1, 7:] = 0  # 7 valid tokens
    response_mask[2, 6:] = 0  # 6 valid tokens
    response_mask[3, 8:] = 0  # 8 valid tokens
    
    index = np.array(['A', 'A', 'B', 'B'], dtype=object)
    traj_index = np.array(['traj_0', 'traj_1', 'traj_2', 'traj_3'], dtype=object)
    
    advantages, returns = compute_advanced_grpo_outcome_advantage(
        episode_rewards=episode_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        norm_adv_by_std_in_grpo=True,
    )
    
    # Verify zero-sum property still holds with varying lengths
    group_a_sum = sum([advantages[i, response_mask[i] > 0].sum().item() for i in [0, 1]])
    group_b_sum = sum([advantages[i, response_mask[i] > 0].sum().item() for i in [2, 3]])
    
    assert abs(group_a_sum) < 1e-4, f"Group A sum should be ~0, got {group_a_sum}"
    assert abs(group_b_sum) < 1e-4, f"Group B sum should be ~0, got {group_b_sum}"
    
    # Verify masked tokens have zero advantage
    assert torch.allclose(advantages[0, 5:], torch.zeros(5), atol=1e-5)
    assert torch.allclose(advantages[1, 7:], torch.zeros(3), atol=1e-5)
    
    print("✓ Varying lengths test passed")


def test_advanced_grpo_numerical_example():
    """Test with a concrete numerical example to verify calculations."""
    batch_size = 2
    response_length = 3
    
    # Episode 1: reward = 1.0, Episode 2: reward = 2.0
    episode_rewards = np.array([1.0, 2.0], dtype=np.float32)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.int64)
    index = np.array(['A', 'A'], dtype=object)
    traj_index = np.array(['traj_0', 'traj_1'], dtype=object)
    
    advantages, returns = compute_advanced_grpo_outcome_advantage(
        episode_rewards=episode_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        norm_adv_by_std_in_grpo=True,
    )
    
    # Manual calculation:
    # Total tokens: 6 (3 per episode)
    # Token rewards: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    # Token mean: 1.5
    # Token std: 0.5
    # Episode 1 token advantages: (1.0 - 1.5) / 0.5 = -1.0
    # Episode 2 token advantages: (2.0 - 1.5) / 0.5 = 1.0
    
    expected_adv_ep0 = -1.0
    expected_adv_ep1 = 1.0
    
    assert torch.allclose(advantages[0, 0], torch.tensor(expected_adv_ep0), atol=1e-5)
    assert torch.allclose(advantages[1, 0], torch.tensor(expected_adv_ep1), atol=1e-5)
    
    # Verify sum is zero
    total_sum = advantages.sum().item()
    assert abs(total_sum) < 1e-4, f"Total sum should be ~0, got {total_sum}"
    
    print("✓ Numerical example verified")


def test_advanced_grpo_without_normalization():
    """Test Advanced GRPO without std normalization."""
    batch_size = 2
    response_length = 5
    
    episode_rewards = np.array([1.0, 2.0], dtype=np.float32)
    response_mask = torch.ones(batch_size, response_length, dtype=torch.int64)
    index = np.array(['A', 'A'], dtype=object)
    traj_index = np.array(['traj_0', 'traj_1'], dtype=object)
    
    advantages, returns = compute_advanced_grpo_outcome_advantage(
        episode_rewards=episode_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        norm_adv_by_std_in_grpo=False,  # Disable std normalization
    )
    
    # Without std normalization:
    # Token mean = 1.5
    # Episode 1 advantages: 1.0 - 1.5 = -0.5
    # Episode 2 advantages: 2.0 - 1.5 = 0.5
    
    assert torch.allclose(advantages[0, 0], torch.tensor(-0.5), atol=1e-5)
    assert torch.allclose(advantages[1, 0], torch.tensor(0.5), atol=1e-5)
    
    # Sum should still be zero
    total_sum = advantages.sum().item()
    assert abs(total_sum) < 1e-4
    
    print("✓ Without normalization test passed")


if __name__ == "__main__":
    test_advanced_grpo_zero_sum_property()
    test_advanced_grpo_token_level_normalization()
    test_advanced_grpo_with_varying_lengths()
    test_advanced_grpo_numerical_example()
    test_advanced_grpo_without_normalization()
    print("\n✓✓✓ All Advanced GRPO tests passed! ✓✓✓")
    print("\nKey property verified: Token-level advantages sum to zero within each group!")

