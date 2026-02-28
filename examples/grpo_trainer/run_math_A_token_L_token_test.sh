set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODE=${1:-train}
if [ "$MODE" == "eval" ] || [ "$MODE" == "evaluation" ]; then
    echo "Running in evaluation mode"
    VAL_ONLY=True
    TRAIN_DATA="$HOME/data/verl_agent_math/train.parquet"
    VAL_DATA="$HOME/data/verl_agent_math/test.parquet" # Full test dataset
    train_data_size=32
    val_data_size=64
    val_group_size=16  # For pass@16 and avg@16 computation during evaluation
else
    echo "Running in training mode"
    VAL_ONLY=False
    TRAIN_DATA="$HOME/data/verl_agent_math/train.parquet"
    VAL_DATA="$HOME/data/verl_agent_math/test_sampled.parquet" # For fast validation during training (test_sampled.parquet contains 50 examples from MATH500, 30 examples from AIME2024, and 30 examples from AIME2025)
    train_data_size=32
    val_data_size=128
    val_group_size=1
fi

group_size=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=math \
    env.seed=0 \
    env.rollout.n=$group_size \
    env.rollout.val_n=$val_group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_math' \
    trainer.experiment_name='grpo_qwen3_4b_A_token_L_token_8k_response_4bsz_1tp' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=300 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    trainer.val_before_train=True \
    trainer.val_only=$VAL_ONLY \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=/home/users/astar/cfar/yux/scratch/zhenglin/verl-agent/checkpoints/verl_agent_math/grpo_qwen3_4b_A_token_L_token_8k_response_4bsz_1tp/global_step_1500
