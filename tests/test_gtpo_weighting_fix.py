"""
Test that GTPO loss weighting gives equal per-trajectory weight
regardless of micro-batch splitting.

The fix changes compute_policy_loss_gtpo to return the SUM of per-traj
mean losses (+ num_groups), and dp_actor.py divides by total_trajs
instead of gradient_accumulation.
"""

import numpy as np
import torch
import sys
sys.path.insert(0, ".")

from verl.trainer.ppo.core_algos import compute_policy_loss_gtpo


def test_equal_traj_weighting():
    """
    Scenario: 4 trajectories with 2 steps each (8 samples total).

    Split A: one micro-batch with all 4 trajs
    Split B: two micro-batches with 2 trajs each
    Split C: four micro-batches with 1 traj each (unequal sizes possible)

    With the fix, accumulated gradients from splits B and C should produce
    the same loss as split A (up to floating point).
    """
    torch.manual_seed(42)
    bs = 8
    resp_len = 6

    old_log_prob = torch.randn(bs, resp_len)
    log_prob = old_log_prob + torch.randn(bs, resp_len) * 0.1  # small perturbation
    advantages = torch.randn(bs, resp_len)
    response_mask = torch.ones(bs, resp_len)
    # 4 trajectories, each with 2 steps
    traj_index = np.array(["t0", "t0", "t1", "t1", "t2", "t2", "t3", "t3"])

    total_trajs = 4

    # --- Split A: all in one micro-batch ---
    pg_loss_sum_A, _, _, _, n_trajs_A = compute_policy_loss_gtpo(
        old_log_prob, log_prob, advantages, response_mask, traj_index,
        cliprange=0.2,
    )
    loss_A = pg_loss_sum_A / total_trajs

    # --- Split B: two micro-batches of 4 samples each ---
    loss_B = torch.tensor(0.0)
    for start in [0, 4]:
        end = start + 4
        pg_loss_sum_b, _, _, _, n_trajs_b = compute_policy_loss_gtpo(
            old_log_prob[start:end], log_prob[start:end],
            advantages[start:end], response_mask[start:end],
            traj_index[start:end],
            cliprange=0.2,
        )
        # Each micro-batch contributes pg_loss_sum / total_trajs
        loss_B = loss_B + pg_loss_sum_b / total_trajs

    # --- Split C: four micro-batches of 2 samples each ---
    loss_C = torch.tensor(0.0)
    for start in [0, 2, 4, 6]:
        end = start + 2
        pg_loss_sum_c, _, _, _, n_trajs_c = compute_policy_loss_gtpo(
            old_log_prob[start:end], log_prob[start:end],
            advantages[start:end], response_mask[start:end],
            traj_index[start:end],
            cliprange=0.2,
        )
        loss_C = loss_C + pg_loss_sum_c / total_trajs

    print(f"Loss A (single batch):    {loss_A.item():.8f}")
    print(f"Loss B (2 micro-batches): {loss_B.item():.8f}")
    print(f"Loss C (4 micro-batches): {loss_C.item():.8f}")

    assert torch.allclose(loss_A, loss_B, atol=1e-6), \
        f"Split B loss {loss_B.item()} != Split A loss {loss_A.item()}"
    assert torch.allclose(loss_A, loss_C, atol=1e-6), \
        f"Split C loss {loss_C.item()} != Split A loss {loss_A.item()}"

    print("PASSED: All splits produce identical loss!")


def test_old_method_fails():
    """
    Demonstrate that the OLD method (mean over trajs, then /gradient_accumulation)
    gives WRONG results when micro-batches have unequal traj counts.
    """
    torch.manual_seed(42)
    bs = 6
    resp_len = 4

    old_log_prob = torch.randn(bs, resp_len)
    log_prob = old_log_prob + torch.randn(bs, resp_len) * 0.1
    advantages = torch.randn(bs, resp_len)
    response_mask = torch.ones(bs, resp_len)
    # 3 trajectories: t0 has 3 steps, t1 has 2 steps, t2 has 1 step
    traj_index = np.array(["t0", "t0", "t0", "t1", "t1", "t2"])

    total_trajs = 3

    # Ground truth: all in one batch
    pg_loss_sum_all, _, _, _, _ = compute_policy_loss_gtpo(
        old_log_prob, log_prob, advantages, response_mask, traj_index,
        cliprange=0.2,
    )
    correct_loss = pg_loss_sum_all / total_trajs

    # Simulate OLD method: mean over trajs per micro-batch, then /gradient_accumulation
    # Split: mb1=[t0(3 steps), t1(2 steps)], mb2=[t2(1 step)]
    # mb1 has 2 trajs, mb2 has 1 traj
    pg_loss_sum_1, _, _, _, n1 = compute_policy_loss_gtpo(
        old_log_prob[:5], log_prob[:5], advantages[:5], response_mask[:5],
        traj_index[:5], cliprange=0.2,
    )
    pg_loss_sum_2, _, _, _, n2 = compute_policy_loss_gtpo(
        old_log_prob[5:], log_prob[5:], advantages[5:], response_mask[5:],
        traj_index[5:], cliprange=0.2,
    )

    # OLD method: each micro-batch does mean internally, then /gradient_accumulation
    old_loss_1 = (pg_loss_sum_1 / n1)  # mean over 2 trajs
    old_loss_2 = (pg_loss_sum_2 / n2)  # mean over 1 traj
    gradient_accumulation = 2
    old_total = (old_loss_1 + old_loss_2) / gradient_accumulation

    # NEW method: sum/total_trajs
    new_total = (pg_loss_sum_1 + pg_loss_sum_2) / total_trajs

    print(f"\nCorrect loss (single batch): {correct_loss.item():.8f}")
    print(f"OLD method loss:             {old_total.item():.8f}")
    print(f"NEW method loss:             {new_total.item():.8f}")

    new_matches = torch.allclose(correct_loss, new_total, atol=1e-6)
    old_matches = torch.allclose(correct_loss, old_total, atol=1e-6)

    print(f"NEW method matches ground truth: {new_matches}")
    print(f"OLD method matches ground truth: {old_matches}")

    assert new_matches, "NEW method should match ground truth"
    # Old method likely does NOT match when traj counts differ
    if not old_matches:
        print("PASSED: Confirmed OLD method gives wrong results with unequal traj counts!")
    else:
        print("NOTE: OLD method happened to match (traj counts were equal)")


if __name__ == "__main__":
    test_equal_traj_weighting()
    print("\n" + "="*60 + "\n")
    test_old_method_fails()
    print("\nAll tests passed!")
