#!/bin/bash
#SBATCH --nodelist=nlpgpu01
#SBATCH --gres=gpu:1 
#SBATCH --time=3-0

# Set up cache directories.
export TRITON_CACHE_DIR=/mnt/nlpgridio3/data/anirudh2/triton_cache
mkdir -p $TRITON_CACHE_DIR

export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/anirudh2/huggingface_data/
export OPENAI_API_KEY=

# Change to your working directory.
cd /mnt/nlpgridio3/data/anirudh2/

OUTPUT_PATH=data/reward_bench_structure_labels.jsonl

python -u main/rewardbench_labeling.py \
  --output_path=${OUTPUT_PATH}