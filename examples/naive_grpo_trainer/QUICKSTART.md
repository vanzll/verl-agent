# Naive GRPO Quick Start

## TL;DR

Naive GRPO computes advantages at the **episode level** instead of the **step level**, then broadcasts them to all steps and tokens.

## Quick Comparison

| Feature | Standard GRPO | Naive GRPO |
|---------|---------------|------------|
| Input | Token-level rewards | Episode rewards |
| Aggregation | Sum tokens → normalize steps | Normalize episodes directly |
| Advantage scope | Per step | Per episode |
| Broadcasting | Step → Tokens | Episode → Steps → Tokens |

## How to Use

### 1. Change One Line in Your Config

Replace:
```bash
algorithm.adv_estimator=grpo
```

With:
```bash
algorithm.adv_estimator=naive_grpo
```

### 2. Run Training

```bash
# AlfWorld example
bash examples/naive_grpo_trainer/run_alfworld_naive_grpo.sh

# Or any other task
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=naive_grpo \
    # ... your other configs
```

That's it! 🎉

## What Changed Under the Hood?

### Before (Standard GRPO)
```python
# Step 1: Episode reward → each step
episode_reward = 10.0  # broadcast to all steps

# Step 2: Sum tokens in each step
step_1_score = sum(tokens_step_1)  # e.g., 10.0
step_2_score = sum(tokens_step_2)  # e.g., 10.0
step_3_score = sum(tokens_step_3)  # e.g., 10.0

# Step 3: Normalize across steps
adv_step_1 = (step_1_score - mean([10,10,10])) / std
adv_step_2 = (step_2_score - mean([10,10,10])) / std
adv_step_3 = (step_3_score - mean([10,10,10])) / std

# Step 4: Broadcast to tokens
all_tokens_step_1 = adv_step_1
all_tokens_step_2 = adv_step_2
all_tokens_step_3 = adv_step_3
```

### After (Naive GRPO)
```python
# Step 1: Normalize episode directly
episode_reward = 10.0
adv_episode = (10.0 - mean_group) / std_group

# Step 2: Broadcast to all steps
adv_step_1 = adv_episode
adv_step_2 = adv_episode
adv_step_3 = adv_episode

# Step 3: Broadcast to all tokens
all_tokens_step_1 = adv_episode
all_tokens_step_2 = adv_episode
all_tokens_step_3 = adv_episode
```

Result: **All tokens in an episode get the same advantage value.**

## When to Use?

✅ **Use Naive GRPO if:**
- Your task has sparse rewards (only at episode end)
- Episode success is what matters most
- You want simpler, more stable training
- All steps should contribute equally

❌ **Use Standard GRPO if:**
- You have dense step-level rewards
- Different steps need different priorities
- You want finer-grained optimization

## Configuration Options

```bash
# Required
algorithm.adv_estimator=naive_grpo

# Optional (with defaults)
algorithm.norm_adv_by_std_in_grpo=True    # Normalize by std
env.rollout.n=4                            # Group size for normalization
algorithm.gamma=1.0                        # Discount factor (not used in outcome-based methods)
```

## Verify It's Working

Add a print statement in training to check:

```python
# In verl/trainer/ppo/core_algos.py, line ~246
print(f"Episode advantage sample: {advantages_scalar[0]}")
print(f"Token advantages (should all be same): {advantages[0, :5]}")
```

You should see the same value repeated across all tokens.

## Example Output

```
Episode advantage sample: 0.8234
Token advantages (should all be same): tensor([0.8234, 0.8234, 0.8234, 0.8234, 0.8234])
```

## Files You Can Safely Ignore

The implementation is self-contained. You only need to:
1. Set `algorithm.adv_estimator=naive_grpo`
2. Run training as usual

Everything else works automatically!

## Need Help?

- **Documentation**: See `examples/naive_grpo_trainer/README.md`
- **Testing**: See `examples/naive_grpo_trainer/TESTING.md`
- **Full details**: See `NAIVE_GRPO_IMPLEMENTATION.md`
- **Example script**: `examples/naive_grpo_trainer/run_alfworld_naive_grpo.sh`

## Credits

Implementation based on the verl-agent framework and GRPO algorithm from DeepSeekMath.

