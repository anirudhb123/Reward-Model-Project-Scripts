#!/bin/bash
#SBATCH --mem=128G        
#SBATCH --nodelist=nlpgpu09
#SBATCH --gres=gpu:1 

source /mnt/nlpgridio3/data/anirudh2/venv/bin/activate
export TRITON_CACHE_DIR=/mnt/nlpgridio3/data/anirudh2/triton_cache
mkdir -p $TRITON_CACHE_DIR

export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/anirudh2/huggingface_data/

cd /mnt/nlpgridio3/data/anirudh2/

TRAIN_EVAL=True
TRAINING_DATA_PATH=data/GEMMA_training_sample.jsonl 
MODEL_NAME=Skywork/Skywork-Reward-Gemma-2-27B-v0.2
python3 -u main/run_fine_tuned_preference_model.py \
--model_name=${MODEL_NAME} \
--train_eval=${TRAIN_EVAL} \
--training_data_path=${TRAINING_DATA_PATH}