# Testing Naive GRPO Implementation

## Running Unit Tests

To run the unit tests for Naive GRPO, make sure you're in the proper Python environment with all dependencies installed:

```bash
# Activate your environment (example)
# conda activate verl-agent
# or
# source venv/bin/activate

# Run the test
cd /home/wanzl/project/verl-agent
python tests/test_naive_grpo.py
```

## Manual Verification

You can also manually verify the implementation by checking the following:

### 1. Check Algorithm Registration

```python
from verl.trainer.ppo.ray_trainer import AdvantageEstimator

# Verify NAIVE_GRPO is registered
assert AdvantageEstimator.NAIVE_GRPO == "naive_grpo"
print("✓ NAIVE_GRPO registered")
```

### 2. Check Core Function

```python
from verl.trainer.ppo.core_algos import compute_naive_grpo_outcome_advantage
import inspect

# Verify function signature
sig = inspect.signature(compute_naive_grpo_outcome_advantage)
params = list(sig.parameters.keys())
assert 'episode_rewards' in params
assert 'response_mask' in params
assert 'index' in params
assert 'traj_index' in params
print("✓ Function signature correct")
```

### 3. Integration Test

Run a small training job with Naive GRPO:

```bash
# Edit the script to use a small model and short training
bash examples/naive_grpo_trainer/run_alfworld_naive_grpo.sh \
    trainer.total_epochs=1 \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    env.max_steps=5
```

### 4. Expected Behavior

When running Naive GRPO, you should observe:

1. **Episode-level advantages**: All steps in the same episode should receive the same advantage value
2. **Group normalization**: Advantages should be normalized across episodes in the same group (uid)
3. **Broadcasting**: The episode-level advantage should be broadcast to all tokens in all steps

You can verify this by adding debug prints in `compute_naive_grpo_outcome_advantage`:

```python
# Add after line 246 in core_algos.py
print(f"Episode rewards: {episode_rewards_tensor[:5]}")  
print(f"Advantages (scalar): {advantages_scalar[:5]}")
print(f"Group means: {[id2mean[idx] for idx in list(set(index[:5]))]}")
```

## Troubleshooting

### Import Error

If you get `ImportError: No module named torch`, ensure:
- You're in the correct Python environment
- Dependencies are installed: `pip install -r requirements.txt`

### Assertion Error in Tests

If tests fail:
1. Check that `episode_rewards` is being passed correctly from the data
2. Verify that `uid` and `traj_uid` are properly set in the rollout loop
3. Ensure `response_mask` has the correct shape

### Runtime Error: No score in prompt index

This means there are no samples for a particular group. Check:
- `env.rollout.n` is set correctly (should be >= 2 for meaningful group normalization)
- Data batching is working correctly
- Trajectory collection is not filtering out all samples

## Comparison with Standard GRPO

To compare Naive GRPO with standard GRPO, run both and compare metrics:

```bash
# Standard GRPO
bash examples/grpo_trainer/run_alfworld.sh

# Naive GRPO  
bash examples/naive_grpo_trainer/run_alfworld_naive_grpo.sh
```

Expected differences:
- Naive GRPO may have more stable gradients (same advantage for entire episode)
- Standard GRPO may learn faster (step-level granularity)
- Final performance depends on the task and reward structure

