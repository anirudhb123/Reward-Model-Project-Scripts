r"""Analyze queries and classify responses into jargon-related categories.

Example usage:

INPUT_PATH=data/GEMMA_training_sample.jsonl
OUTPUT_PATH=data/GEMMA_sample_labeled_jargon.jsonl
python3 main/jargon_labeling.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
--model_name=gpt-4
"""

from absl import app
from absl import flags

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models
from data_utils import jsonl_utils
import tqdm

_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file containing queries and responses."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file with labeled jargon-related classifications."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4", "Model name to use for classification."
)

def main(unused_argv) -> None:
    classification_prompt = """
You are a query classifier. Your task is to classify the following query and responses into three categories:

1. **Query Classification:** First, classify the query into one of the following categories:
   Closed-ended, Open-ended, Opinion-based, Procedural, Comparative, Hypothetical, Technical, Ambiguous, or Adversarial.
   Provide only the category name as the output.

2. **Chosen Response Contains Jargon:** Does the chosen response predominantly consist of jargon? (Yes/No)
   *For this classification, consider a response to predominantly contain jargon only if specialized terms or technical language are used so extensively that they dominate the response and make it largely inaccessible to a non-specialist.*

3. **Rejected Response Contains Jargon:** Does the rejected response predominantly consist of jargon? (Yes/No)
   *Apply the same qualitative standard: only classify as Yes if the response is saturated with specialized language to the point that a general audience would struggle to understand the majority of the content.*

Here are the query and responses:

- **Query:** [QUERY]
- **Chosen Response:** [CHOSEN]
- **Rejected Response:** [REJECTED]

Provide your answers in the following format:
Query Classification: [Classification]
Chosen contains Jargon: [Yes/No]
Rejected contains Jargon: [Yes/No]
"""

    examples = jsonl_utils.read(_INPUT_PATH.value)

    if "gpt" in _MODEL_NAME.value:
        model = models.GPT4(model_name=_MODEL_NAME.value)
    elif "gemini" in _MODEL_NAME.value:
        model = models.Gemini(model_name=_MODEL_NAME.value)
    elif "claude" in _MODEL_NAME.value:
        model = models.Claude(model_name=_MODEL_NAME.value)
    elif "jamba" in _MODEL_NAME.value:
        model = models.Jamba(model_name=_MODEL_NAME.value)
    else:
        model = models.TogetherAI(model_name=_MODEL_NAME.value)

    outputs = []
    for idx, ex in enumerate(tqdm.tqdm(examples[:2500])):
        cur_prompt = (
            classification_prompt
            .replace("[QUERY]", ex.get("query", ""))
            .replace("[CHOSEN]", ex.get("chosen", ""))
            .replace("[REJECTED]", ex.get("rejected", ""))
        )

        generated_output = model.generate(input_text=cur_prompt, max_len=100).strip()

        try:
            parsed_lines = [line.split(": ")[1].strip() for line in generated_output.split("\n") if ": " in line]
            asked_for_jargon = parsed_lines[0] if len(parsed_lines) > 0 else ""
            chosen_jargon = parsed_lines[1] if len(parsed_lines) > 1 else ""
            rejected_jargon = parsed_lines[2] if len(parsed_lines) > 2 else ""
        except Exception as e:
            asked_for_jargon, chosen_jargon, rejected_jargon = "", "", ""

        outputs.append({
            "asked_for_jargon": asked_for_jargon,
            "chosen_jargon": chosen_jargon,
            "rejected_jargon": rejected_jargon
        })

        if (idx + 1) % 50 == 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)

        if len(outputs) % 50 != 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
    app.run(main)

