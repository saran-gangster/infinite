#!/bin/bash
# GRPO Training Launch Script for the verifiers-integrated trainer.

# Setup Hugging Face environment to handle rate limiting
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"

# Load environment variables if .env file exists
if [ -f "env/.env" ]; then
    echo "Loading environment variables from env/.env"
    source env/.env
fi

# Check for HF token
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "WARNING: No Hugging Face token found! Set HF_TOKEN to avoid rate limiting."
    sleep 3
else
    echo "Using Hugging Face token (${HF_TOKEN:0:8}...)"
fi

echo "Starting GRPO training with multi-domain verifiers environments..."
echo ""

torchrun \
    --nproc_per_node=8 \
    -m train.trainer.grpo \
    --config-name grpo \
    actor.model_name=Qwen/Qwen2.5-7B-Instruct \
    actor.max_length_per_device=4096 \
    data.prompts_per_rollout=64 \
    data.responses_per_prompt=16 \
    trainer.project=Infinite-Verifiers \
    trainer.experiment_name=qwen-7b-math-wordle