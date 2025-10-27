# Advanced GRPO Trainer

## Overview

**Advanced GRPO** is a token-level normalization variant of GRPO that ensures all token-level advantages within a group sum to zero.

## Core Idea

### The Problem Advanced GRPO Solves

In standard GRPO and Naive GRPO, advantages are computed at the episode or step level, then broadcast to tokens. This means:
- The sum of advantages across episodes/steps in a group is zero
- But the sum of advantages across **all tokens** in a group is **not necessarily zero**

Advanced GRPO fixes this by normalizing at the token level.

## Algorithm Flow

```
1. Episode Reward (scalar) 
   ↓
2. Broadcast to all steps in episode
   ↓
3. Broadcast to all tokens in each step
   ↓
4. Collect all tokens in the same group (uid)
   ↓
5. Compute token-level mean and std across all group tokens
   ↓
6. Normalize each token: adv = (token_reward - token_mean) / token_std
```

## Mathematical Formulation

### Advanced GRPO

For group `g` with episodes `E = {e₁, e₂, ...}`:

```
1. For each token t in episode e:
   token_reward[t] = episode_reward[e]

2. Collect all tokens in group g:
   T_g = {all tokens from all episodes in g}

3. Compute token-level statistics:
   μ_tokens = mean(token_reward[t] for t in T_g)
   σ_tokens = std(token_reward[t] for t in T_g)

4. Normalize each token:
   adv[t] = (token_reward[t] - μ_tokens) / σ_tokens
```

**Key Property**: `Σ_{t ∈ T_g} adv[t] = 0` (zero-mean normalization)

## Comparison with Other Methods

| Method | Normalization Level | Sum-to-Zero Guarantee |
|--------|-------------------|---------------------|
| **Standard GRPO** | Step-level | Steps in group |
| **Naive GRPO** | Episode-level | Episodes in group |
| **Advanced GRPO** | Token-level | **All tokens in group** |

## Example

Consider a group with 2 episodes:

**Episode 1**: reward = 1.0, 3 steps × 10 tokens = 30 tokens
**Episode 2**: reward = 2.0, 3 steps × 10 tokens = 30 tokens

### Standard GRPO
```
Step-level scores: [1.0, 1.0, 1.0] for E1, [2.0, 2.0, 2.0] for E2
Step mean = 1.5, std = 0.5
Step advantages: [-1.0, -1.0, -1.0] for E1, [1.0, 1.0, 1.0] for E2
Token advantages: Each step's advantage → all its tokens
Sum of step advantages = 0 ✓
Sum of all token advantages = 30×(-1.0)×3 + 30×(1.0)×3 = -90 + 90 = 0 ✓
(Works because all steps have same length)
```

### Naive GRPO
```
Episode advantages: (1.0-1.5)/0.5 = -1.0 for E1, (2.0-1.5)/0.5 = 1.0 for E2
Token advantages: -1.0 for all E1 tokens, 1.0 for all E2 tokens
Sum of all token advantages = 30×(-1.0) + 30×(1.0) = 0 ✓
(Works because both episodes have same number of tokens)
```

### Advanced GRPO
```
Token rewards: [1.0] × 30 for E1, [2.0] × 30 for E2
All tokens: 60 tokens total
Token mean = (30×1.0 + 30×2.0) / 60 = 1.5
Token std = 0.5
Token advantages: (1.0-1.5)/0.5 = -1.0 for each E1 token
                  (2.0-1.5)/0.5 = 1.0 for each E2 token
Sum of all token advantages = 30×(-1.0) + 30×(1.0) = 0 ✓
(Always works by construction!)
```

## Why Token-Level Normalization?

### Advantages

1. **Guaranteed Zero-Sum**: Token advantages always sum to zero within a group
2. **Fine-Grained**: Each token gets individually normalized
3. **Robust**: Works regardless of episode length variations
4. **Mathematical Property**: Leverages zero-mean normalization directly

### When to Use

✅ **Use Advanced GRPO when:**
- You want strict zero-sum property across all tokens
- Episodes in a group have varying lengths
- You want maximum fine-grained control
- Mathematical elegance matters

❌ **Consider alternatives when:**
- Episodes have very different token counts (may over-weight longer episodes)
- You want episode-level or step-level granularity
- Computational cost is a concern (more tokens to process)

## Usage

### Command Line

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=advanced_grpo \
    # ... other configs
```

### Example Script

```bash
bash examples/advanced_grpo_trainer/run_alfworld_advanced_grpo.sh
```

### Configuration

```yaml
algorithm:
  adv_estimator: advanced_grpo
  norm_adv_by_std_in_grpo: true  # Recommended: true
```

## Implementation Details

**File**: `verl/trainer/ppo/core_algos.py`

```python
def compute_advanced_grpo_outcome_advantage(
    episode_rewards,     # Episode-level rewards
    response_mask,       # Token validity mask
    index,              # Group ID (uid)
    traj_index,         # Trajectory ID
    ...
):
    # 1. Broadcast episode reward to all tokens
    token_rewards = episode_reward.expand(all_tokens)
    
    # 2. Collect all tokens per group
    group_tokens = collect_by_group(token_rewards, index)
    
    # 3. Compute token-level statistics
    token_mean = mean(group_tokens)
    token_std = std(group_tokens)
    
    # 4. Normalize each token
    advantages = (token_rewards - token_mean) / token_std
```

## Theoretical Properties

### Zero-Sum Guarantee

For any group `g`:
```
Σ_{t ∈ group_g} advantage[t] = 0
```

**Proof**: By construction, advantages are computed as `(x - μ) / σ` where `μ` is the mean of all tokens in the group. Therefore, the sum of all `(x - μ)` terms is zero, and dividing by `σ` preserves this property.

### Variance

Token-level normalization may result in:
- **Lower variance** if episode rewards are similar (many tokens get similar values)
- **Higher variance** if episode rewards differ greatly (tokens from different episodes get very different values)

## Comparison Summary

| Aspect | Standard GRPO | Naive GRPO | Advanced GRPO |
|--------|---------------|------------|---------------|
| **Input** | Token-level rewards | Episode rewards | Episode rewards |
| **Aggregation** | Sum tokens/step | Episode direct | Episode → tokens |
| **Normalization** | Step-level | Episode-level | **Token-level** |
| **Zero-sum** | Steps in group | Episodes in group | **All tokens in group** |
| **Granularity** | Step | Episode | Token |
| **Robustness** | Good | Good | **Best** |

## Tips and Tricks

1. **Group Size**: Use `env.rollout.n >= 2` for meaningful normalization
2. **Token Count**: Works best when episodes have reasonable token counts
3. **Debugging**: Check that `sum(advantages[group])` ≈ 0 for each group
4. **Monitoring**: Track token-level statistics in logs

## Future Work

Potential enhancements:
- Weighted token normalization (weight by position or importance)
- Hybrid token-step normalization
- Adaptive normalization based on token count

---

**Status**: ✅ Implemented and ready to use

**Key Feature**: Token-level advantages guaranteed to sum to zero within each group

