#!/bin/bash

export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/cmalaviya/huggingface_data/

cd /mnt/nlpgridio3/data/cmalaviya/rlhf-bias

source set_keys.sh

INPUT_PATH=data/all_data.jsonl
OUTPUT_PATH=data/all_data_baseline.jsonl
python3 main/generate_base_responses.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
