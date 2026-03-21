# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**verl-agent** extends [veRL](https://github.com/volcengine/verl) (Volcano Engine RL for LLM) with multi-turn agent training via reinforcement learning. Its key contribution is the **GiGPO** (Group-in-Group Policy Optimization) algorithm (NeurIPS 2025, arXiv:2505.10978).

## Common Commands

```bash
# Install (editable)
pip3 install -e .

# Lint (ruff, line-length=300)
ruff check --fix .
ruff format .

# Run tests
pytest tests/

# Data preprocessing
python3 -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 128

# Training (main entry point)
python3 -m verl.trainer.main_ppo <hydra overrides>

# Example training scripts
bash examples/gigpo_trainer/run_alfworld.sh
bash examples/grpo_trainer/run_alfworld_A_Step_L_token.sh
bash examples/ppo_trainer/run_alfworld.sh
```

## Architecture

### Three-Layer Structure

1. **`verl/`** — Core RL training framework (forked from veRL)
   - `verl/protocol.py` — `DataProto`: the central data exchange format wrapping TensorDict + non-tensor data, used everywhere
   - `verl/trainer/main_ppo.py` — Main Hydra entry point, initializes Ray, envs, tokenizer, trainer
   - `verl/trainer/ppo/ray_trainer.py` — `RayPPOTrainer`: core training loop, advantage estimation dispatch, worker orchestration
   - `verl/trainer/ppo/core_algos.py` — RL algorithm implementations (GAE, GRPO, REINFORCE++, KL controllers)
   - `verl/trainer/config/ppo_trainer.yaml` — Default Hydra config with all hyperparameters
   - `verl/workers/` — Distributed workers (Actor, Rollout, Critic, RewardModel) with FSDP and Megatron backends
   - `verl/single_controller/` — Ray-based single-controller distributed orchestration

2. **`agent_system/`** — Agent-environment interaction layer
   - `agent_system/environments/base.py` — `EnvironmentManagerBase` ABC with `reset/step/build_text_obs/success_evaluator`
   - `agent_system/environments/env_manager.py` — Concrete env managers + `make_envs()` factory
   - `agent_system/environments/prompts/` — Per-environment prompt templates
   - `agent_system/multi_turn_rollout/rollout_loop.py` — `TrajectoryCollector`: multi-turn rollout loop, batch assembly
   - `agent_system/memory/memory.py` — `SimpleMemory`/`SearchMemory` for history management
   - `agent_system/reward_manager/episode.py` — `EpisodeRewardManager` for episode-level rewards

3. **`gigpo/`** — GiGPO algorithm implementation
   - `gigpo/core_gigpo.py` — Step-level grouping, anchor state hashing, similarity-based grouping, two-level advantage computation

### Training Pipeline Flow

1. `main_ppo.py` loads Hydra config → initializes Ray → creates envs via `make_envs()` → launches workers
2. `RayPPOTrainer.fit()` loops over data batches
3. Multi-turn rollout via `TrajectoryCollector.vanilla_multi_turn_loop()`: env reset → generate action → env.step() → collect rewards
4. Advantage estimation dispatched by `algorithm.adv_estimator` config: `gae`, `grpo`, `gigpo`, `rloo`, `reinforce_plus_plus`, etc.
5. Actor update (PPO clip), optional critic update, validation, wandb logging

### Supported Environments

ALFWorld, WebShop, Search (Search-R1), Sokoban, Gym Cards (EZPoints, Points24, NumberLine, Blackjack), AppWorld, Math. Register new ones in `make_envs()` by subclassing `EnvironmentManagerBase`.

## Key Design Decisions

- **Grouping**: `actor_rollout_ref.rollout.n` must be 1 in verl-agent; group size is controlled by `env.rollout.n` instead (multiple rollouts from same initial state)
- **Step-independent rollout**: Each step's input is independently constructed (not concatenated history), enabling scalability for long-horizon tasks
- **Hybrid engine**: Actor and Rollout share the same worker for memory efficiency
- **Config**: Hydra + OmegaConf variable interpolation (e.g., `${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}`)
- **Distributed**: Supports both FSDP/FSDP2 and Megatron distributed strategies, orchestrated via Ray
- **LoRA**: Supported for parameter-efficient fine-tuning
- **GTPO training loop (Plan B)**: GTPO with `pure_on_policy=True` collapses all mini-batches into 1 full-batch (`ppo_mini_batch_size = batch_length`) and runs a fixed number of epochs (default 4; override via `ppo_epochs > 1`). Each epoch = 1 optimizer step on the full batch. This avoids the instability of the stochastic mini-batch approach (Plan A, preserved on `plan-A` branch) which created ~33 sequential optimizer steps per iteration, causing KL explosion (0.2+) and gradient norm chaos (10-15x variance across mini-batches) due to stale `old_log_prob`. The Plan A code is preserved on the `plan-A` branch for reference.
- **GTPO micro-batch & GPU memory**: `split_micro_batches_by_trajectory` keeps entire trajectories intact within micro-batches. The **effective micro-batch size is bounded by the longest trajectory** (up to `env.max_steps`), regardless of `ppo_micro_batch_size_per_gpu`. The dominant GPU memory consumer is the logits tensor `(total_nnz × vocab_size)` — for Qwen2.5 (vocab=151,936) this can be 15–20 GiB, multiplied by 2–3× during forward.
- **GTPO loss normalization**: GTPO's `pg_loss` is normalized via `total_trajs_in_minibatch` weighting (not `gradient_accumulation` division like vanilla/GSPO). Auxiliary losses (entropy, KL) are micro-batch-level means and must be explicitly divided by `gradient_accumulation` in the GTPO branch.
- **vLLM KV cache & GPU memory fragmentation**: `gpu_memory_utilization` controls how much free GPU memory vLLM pre-allocates for KV cache during rollout. With `free_cache_engine=True`, the cache is released after rollout, but without `gc.collect()` + `torch.cuda.empty_cache()` the memory stays in PyTorch's CUDA allocator pool — reserved but potentially fragmented. This can cause OOM during training when large contiguous tensors (e.g., logits) are needed. The safe default is `gpu_memory_utilization=0.5`; values ≥0.6 risk OOM on tight-memory setups. Note that `expandable_segments` is temporarily disabled during vLLM initialization (`vllm_rollout_spmd.py`), so it doesn't help with KV cache fragmentation.

## Dependencies

PyTorch 2.6.0, transformers (<=4.51.1), Ray (>=2.41.0,<2.50.0), vLLM (<=0.8.5) or SGLang, flash-attn, Hydra, wandb. Python 3.12 recommended.


### Developer Tips
- 每次回答请以 Hi Carlos 开始
- 任何我希望你修改代码的地方， 请你都用以下工作流：
  - 先本地main分支 pull 远程main分支的最新代码
  - 然后新建一个git worktree， 使用你的coding 相关的skills修改代码push到远程并提draft pr， 完成后使用 Codex skill (`skill-codex:codex`) review你的代码并在PR上提comment，直到代码没有问题之后把pr从draft 变成ready for review， 然后你需要等待远程reviewer的操作： 提comments 或者 merge， 如果是comments， 那么就根据comments 继续修改pr，如果是merge， 则删除本地worktree并pull 远程main到本地main。
    - Tips: 修改代码时，你大概率需要用到superpowers里面和写代码，软件开发相关的skill，并思考你是否需要用到其他coding相关的skill
    - Codex review 规则: 如果当前对话 session 中已经调用过 Codex，后续再次调用时必须使用 `codex exec resume --last` 来延续同一个上下文窗口，保持 Codex 的完整上下文不丢失。同时在 resume 的 prompt 中提供足够的上下文信息（如 PR 链接、改动摘要、需要关注的重点），让 Codex 能够高效地继续工作。
    - Codex 模型选择: 使用gpt5.3-codex 模型，并使用高 reasoning effort（high），不需要每次询问用户。
- 我对这个项目的文档是/home/wanzl/project/Agentic-World-Model/agentic-world-model/c_knowledge.md .在任务过程中有任何你觉得对于项目有实质性帮助的东西，都请添加到项目根目录的c_knowledge.md里面。如果没有特别的指令，CLAUDE.md需要被git 追踪。


### WANDB LINK
Alfworld 1.5B: https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/verl_agent_alfworld
Webshop 1.5B: https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/verl_agent_webshop
Math 4B: https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/verl_agent_math?nw=nwuservanzl3386
你必须关注有效的run。（有效的： 指我网页端的过滤器过滤之后还保留的， 以及是“显示”状态的， 无效的： 指被网页端过滤器过滤掉的， 以及被我手动设置成不显示的。）

### 飞书云文档
Loss-Advantage Mismatch: https://my.feishu.cn/docx/CWmadoODHongjQxCS9nc32rinnb (document_id: CWmadoODHongjQxCS9nc32rinnb) 
