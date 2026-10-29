#!/usr/bin/env bash
# Shared MPO pipeline launcher. Run the public scripts below from any directory.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the policy model directory}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/bcp}"
TRAIN_FILE="$DATA_DIR/train.parquet"
VAL_FILE="$DATA_DIR/validation.parquet"
TOOL_CONFIG_PATH="$PROJECT_DIR/configs/search_tools.yaml"
export RAG_URL="${RAG_URL:-http://localhost:8000}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONUNBUFFERED=1
export RAY_USAGE_STATS_ENABLED=0 RAY_enable_open_telemetry=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
export VERL_MASTER_PORT_RANGE="${VERL_MASTER_PORT_RANGE:-41000,42000}"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
NNODES="${N_NODE:-1}"
N_GPU_PER_NODE="${N_GPU_PER_NODE:-8}"
ACTOR_TP="${ACTOR_TP:-2}"; ACTOR_PP="${ACTOR_PP:-1}"; ACTOR_CP="${ACTOR_CP:-1}"
CRITIC_TP="$ACTOR_TP"; CRITIC_PP="$ACTOR_PP"; CRITIC_CP="$ACTOR_CP"
REF_TP="$ACTOR_TP"; REF_PP="$ACTOR_PP"; REF_CP="$ACTOR_CP"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
N_RESP="${N_RESP:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-16000}"
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-5}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-20}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-32768}"
TOKEN_BUDGET=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_TOKEN_BUDGET_PER_GPU=$(((TOKEN_BUDGET + ACTOR_CP - 1) / ACTOR_CP))
REF_TOKEN_BUDGET_PER_GPU=$ACTOR_TOKEN_BUDGET_PER_GPU
CRITIC_TOKEN_BUDGET_PER_GPU=$ACTOR_TOKEN_BUDGET_PER_GPU
ADV_ESTIMATOR=gae; GAE_GAMMA=1.0
GAE_LAM="${GAE_LAM:-0.8}"
INTRA_TURN_LAM="${INTRA_TURN_LAM:-0.95}"
INTRA_TURN_BETA="${INTRA_TURN_BETA:-1.0}"
TURN_LEVEL_LAM=True; TURN_VALUE_MODE=first_assistant
CRITIC_LR="${CRITIC_LR:-1e-5}"
CRITIC_PPO_EPOCHS="${CRITIC_PPO_EPOCHS:-1}"
CLIPRANGE_VALUE="${CLIPRANGE_VALUE:-1.0}"
CRITIC_MULTI_TURN_MASK=True
CRITIC_WARMUP=0; CRITIC_WARMUP_SAVE=False; VALUE_HEAD_ZERO_INIT=False
CRITIC_CHECKPOINT_LOAD_CONTENTS="${CRITIC_CHECKPOINT_LOAD_CONTENTS:-[model,optimizer]}"
CRITIC_INIT_FROM_PATH="${CRITIC_INIT_FROM_PATH:-}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
STAGE_OVERRIDES=()
case "${MPO_STAGE:-train}" in
    value_pretrain)
        GAE_LAM=1.0; INTRA_TURN_BETA=0.0
        TURN_LEVEL_LAM=False; TURN_VALUE_MODE=none
        TOTAL_TRAINING_STEPS="${VP_STEPS:-50}"
        CRITIC_WARMUP=$((TOTAL_TRAINING_STEPS + 1))
        CRITIC_WARMUP_SAVE=True; VALUE_HEAD_ZERO_INIT=True
        CRITIC_INIT_FROM_PATH=""
        STAGE_OVERRIDES+=(trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=10)
        ;;
    train)
        : "${CRITIC_INIT_FROM_PATH:?Set CRITIC_INIT_FROM_PATH to the value-pretrain global_step_N directory}"
        ;;
    validate)
        : "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an MPO training global_step_N directory}"
        CRITIC_INIT_FROM_PATH=""
        # This reuses training's rollout, tool formatting and strict BCP reward.
        STAGE_OVERRIDES+=(trainer.val_only=True trainer.val_before_train=True
            trainer.resume_mode=resume_path "trainer.resume_from_path=$CHECKPOINT_PATH"
            'actor_rollout_ref.actor.checkpoint.load_contents=[model]'
            'critic.checkpoint.load_contents=[model]'
            actor_rollout_ref.rollout.val_kwargs.n=1
            actor_rollout_ref.rollout.val_kwargs.do_sample=False)
        ;;
    *) printf 'Unknown MPO_STAGE\n' >&2; exit 2 ;;
esac
TOTAL_EPOCHS="${TOTAL_EPOCHS:-$TOTAL_TRAINING_STEPS}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mpo_${MPO_STAGE:-train}}"
CKPT_DIR="${RUN_LOCAL_DIR:-$PROJECT_DIR/checkpoints/$EXPERIMENT_NAME}"
VALIDATION_DATA_DIR="$PROJECT_DIR/outputs/$EXPERIMENT_NAME/validation"
ROLLOUT_DATA_DIR="$PROJECT_DIR/outputs/$EXPERIMENT_NAME/rollouts"
export TENSORBOARD_DIR="$PROJECT_DIR/tensorboard_log/$EXPERIMENT_NAME"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    for path in "$MODEL_PATH/config.json" "$TRAIN_FILE" "$VAL_FILE"; do
        [[ -f "$path" ]] || { printf 'Missing required file: %s\n' "$path" >&2; exit 1; }
    done
    if [[ "${MPO_STAGE:-train}" == train ]]; then
        [[ -d "$CRITIC_INIT_FROM_PATH/critic" ]] || { printf 'Missing value-pretrain critic checkpoint\n' >&2; exit 1; }
    fi
    if [[ "${MPO_STAGE:-train}" == validate ]]; then
        [[ -d "$CHECKPOINT_PATH/actor" && -d "$CHECKPOINT_PATH/critic" ]] || { printf 'Missing actor/critic checkpoint\n' >&2; exit 1; }
    fi
    "$PYTHON_BIN" -m mpo.preflight --nodes "$NNODES"
    mkdir -p "$CKPT_DIR" "$VALIDATION_DATA_DIR" "$ROLLOUT_DATA_DIR" "$TENSORBOARD_DIR"
fi
command=("$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-path="$PROJECT_DIR/configs" \
  --config-name=mpo_bcp \
  \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  algorithm.gamma="$GAE_GAMMA" \
  algorithm.lam="$GAE_LAM" \
  algorithm.lam_critic="${GAE_LAM_CRITIC:-1.0}" \
  algorithm.turn_level_lam="$TURN_LEVEL_LAM" \
  algorithm.turn_value_mode="$TURN_VALUE_MODE" \
  algorithm.intra_turn_beta="$INTRA_TURN_BETA" \
  algorithm.intra_turn_lam="$INTRA_TURN_LAM" \
  algorithm.intra_turn_use_reward="${INTRA_TURN_USE_REWARD:-True}" \
  algorithm.intra_turn_first_token_anchor="${INTRA_TURN_FIRST_TOKEN_ANCHOR:-False}" \
  algorithm.use_kl_in_reward=False \
  \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=False \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  +data.tool_config_path="$TOOL_CONFIG_PATH" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_TOKEN_BUDGET_PER_GPU" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio=0.28 \
  actor_rollout_ref.actor.clip_ratio_low=0.20 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing="${USE_DIST_CKPT:-False}" \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$ACTOR_TP" \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="$ACTOR_PP" \
  actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
  actor_rollout_ref.actor.megatron.context_parallel_size="$ACTOR_CP" \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  actor_rollout_ref.actor.megatron.param_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.n="$N_RESP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$ACTOR_TOKEN_BUDGET_PER_GPU" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length="$MAX_MODEL_LEN" \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size="$SGLANG_CHUNKED_PREFILL_SIZE" \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens="$SGLANG_MAX_PREFILL_TOKENS" \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=True \
  \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.format="${MULTI_TURN_FORMAT:-hermes}" \
  +actor_rollout_ref.rollout.multi_turn.inject_tool_schemas=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$REF_TOKEN_BUDGET_PER_GPU" \
  actor_rollout_ref.ref.megatron.use_mbridge=True \
  actor_rollout_ref.ref.megatron.use_dist_checkpointing="${USE_DIST_CKPT:-False}" \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="$REF_TP" \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="$REF_PP" \
  actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=null \
  actor_rollout_ref.ref.megatron.context_parallel_size="$REF_CP" \
  actor_rollout_ref.ref.megatron.param_offload=True \
  \
  critic.enable=True \
  critic.checkpoint.load_contents="$CRITIC_CHECKPOINT_LOAD_CONTENTS" \
  critic.model.path="$MODEL_PATH" \
  critic.optim.lr="$CRITIC_LR" \
  critic.ppo_epochs="$CRITIC_PPO_EPOCHS" \
  critic.cliprange_value="$CLIPRANGE_VALUE" \
  critic.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.use_dynamic_bsz=True \
  critic.ppo_max_token_len_per_gpu="$CRITIC_TOKEN_BUDGET_PER_GPU" \
  critic.megatron.use_mbridge=True \
  critic.megatron.use_dist_checkpointing="${USE_DIST_CKPT:-False}" \
  critic.megatron.tensor_model_parallel_size="$CRITIC_TP" \
  critic.megatron.pipeline_model_parallel_size="$CRITIC_PP" \
  critic.megatron.virtual_pipeline_model_parallel_size=null \
  critic.megatron.context_parallel_size="$CRITIC_CP" \
  critic.megatron.override_transformer_config.recompute_method=uniform \
  critic.megatron.override_transformer_config.recompute_granularity=full \
  critic.megatron.override_transformer_config.recompute_num_layers=1 \
  critic.megatron.param_offload=True \
  critic.megatron.grad_offload=True \
  critic.megatron.optimizer_offload=True \
  ++critic.value_head_zero_init="$VALUE_HEAD_ZERO_INIT" \
  ++critic.critic_multi_turn_mask="$CRITIC_MULTI_TURN_MASK" \
  \
  reward.custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/bc_p_llm_judge.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_model.enable=False \
  \
  +ray_kwargs.ray_init.address=auto \
  +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
  ++ray_kwargs.ray_init.runtime_env.env_vars.RAG_URL="$RAG_URL" \
  ++ray_kwargs.ray_init.runtime_env.env_vars.BCP_JUDGE_ENV_FILE="${BCP_JUDGE_ENV_FILE:-}" \
  \
  ++trainer.critic_init_from_path="$CRITIC_INIT_FROM_PATH" \
  trainer.critic_warmup="$CRITIC_WARMUP" \
  ++trainer.critic_warmup_save="$CRITIC_WARMUP_SAVE" \
  trainer.n_gpus_per_node="${N_GPU_PER_NODE:-8}" \
  trainer.nnodes="$NNODES" \
  trainer.project_name=multiscale-policy-optimization \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.logger='["console","tensorboard"]' \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.test_freq="${TEST_FREQ:-5}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.resume_mode="${RESUME_MODE:-auto}" \
  +trainer.master_port_range='[31000,32000]' \
  +trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  +trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  "${STAGE_OVERRIDES[@]}" \
  "$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
