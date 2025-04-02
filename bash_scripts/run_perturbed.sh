#!/bin/bash

cd /mnt/nlpgridio3/data/anirudh2/
export OPENAI_API_KEY="PLACEHOLDER"
INPUT_PATH=data/all_data_baseline.jsonl
BIAS=elaboration
OUTPUT_PATH=data/all_data_perturbed_${BIAS}.jsonl
python3 main/generate_perturbed_responses.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}