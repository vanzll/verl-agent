# Naive GRPO Trainer

## Overview

This directory contains examples for training agents using **Naive GRPO** (Group Relative Policy Optimization).

## Difference between Standard GRPO and Naive GRPO

### Standard GRPO (Original Implementation)
1. **Episode reward broadcasting**: Broadcasts `episode_rewards` to each step of the trajectory
2. **Step-level advantage computation**: For each step, sums token-level rewards to get a score
3. **Group normalization**: Normalizes scores across steps within the same group (uid)
4. **Token broadcasting**: Broadcasts the normalized score to all tokens in that step

**Formula**: 
```
step_score[i] = sum(token_level_rewards[i, :])
advantage[i] = (step_score[i] - group_mean) / group_std
advantage[i, t] = advantage[i]  # broadcast to all tokens t
```

### Naive GRPO (New Implementation)
1. **Direct episode-level computation**: Uses `episode_rewards` directly (no step-level aggregation)
2. **Episode-level advantage**: Normalizes episode rewards across episodes in the same group
3. **Episode + Step + Token broadcasting**: Broadcasts the episode-level advantage to all steps, then to all tokens

**Formula**:
```
episode_advantage[i] = (episode_rewards[i] - group_mean) / group_std
advantage[i, t] = episode_advantage[i]  # broadcast to all tokens t in all steps of episode i
```

## Key Differences

| Aspect | Standard GRPO | Naive GRPO |
|--------|---------------|------------|
| Advantage Level | Step-level | Episode-level |
| Aggregation | Sum token rewards per step | Use episode reward directly |
| Normalization | Across steps in group | Across episodes in group |
| Broadcasting | Step → Tokens | Episode → Steps → Tokens |

## When to Use Naive GRPO

Use Naive GRPO when:
- You want a simpler, more direct mapping from episode outcomes to optimization
- Episode-level rewards are sparse (only at the end)
- You want all steps in an episode to receive the same optimization signal
- You prefer stronger coupling between episode success and gradient updates

Use Standard GRPO when:
- You have step-level reward signals
- You want more granular optimization at the step level
- Different steps in an episode should potentially receive different advantages

## Usage

To run Naive GRPO on AlfWorld:

```bash
bash examples/naive_grpo_trainer/run_alfworld_naive_grpo.sh
```

The key configuration parameter is:
```bash
algorithm.adv_estimator=naive_grpo
```

## Implementation Details

The Naive GRPO implementation is located in:
- `verl/trainer/ppo/core_algos.py`: `compute_naive_grpo_outcome_advantage()`
- `verl/trainer/ppo/ray_trainer.py`: `AdvantageEstimator.NAIVE_GRPO` case

The function signature:
```python
def compute_naive_grpo_outcome_advantage(
    episode_rewards: np.ndarray,        # Shape: (batch_size,)
    response_mask: torch.Tensor,         # Shape: (batch_size, response_length)
    index: np.ndarray,                   # Group ID (uid)
    traj_index: np.ndarray,              # Trajectory ID
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:  # Returns: (advantages, returns)
```

## Configuration Options

All standard PPO/GRPO options apply, plus:
- `algorithm.adv_estimator=naive_grpo`: Enable Naive GRPO
- `algorithm.norm_adv_by_std_in_grpo=True/False`: Whether to normalize by std (default: True)
- `env.rollout.n`: Number of rollouts per prompt for group normalization

## Citation

If you use Naive GRPO in your research, please cite both the original GRPO paper and this implementation:

```bibtex
@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and others},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```

