r"""Analyze queries and classify responses into elaboration-related categories.

Example usage:

INPUT_PATH=data/GEMMA_training_sample.jsonl
OUTPUT_PATH=data/GEMMA_sample_labeled_elaboration.jsonl
python3 main/elaboration_labeling.py \
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
    "output_path", "", "Path to the output file with labeled elaboration-related classifications."
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

2. **Chosen Response is Excessively Elaborate:** Does the chosen response contain excessive elaboration? (Yes/No)
   *A response should be marked as excessively elaborate if it adds unnecessary details, expands descriptions without improving clarity, or significantly lengthens the response without adding new meaning.*

3. **Rejected Response is Excessively Elaborate:** Does the rejected response contain excessive elaboration? (Yes/No)
   *Apply the same standard: only classify as Yes if the response introduces redundant details or expands unnecessarily while keeping the same meaning.*

Here are the query and responses:

- **Query:** [QUERY]
- **Chosen Response:** [CHOSEN]
- **Rejected Response:** [REJECTED]

Provide your answers in the following format:
Query Classification: [Classification]
Chosen is Excessively Elaborate: [Yes/No]
Rejected is Excessively Elaborate: [Yes/No]
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
            query_classification = parsed_lines[0] if len(parsed_lines) > 0 else ""
            chosen_elaborate = parsed_lines[1] if len(parsed_lines) > 1 else ""
            rejected_elaborate = parsed_lines[2] if len(parsed_lines) > 2 else ""
        except Exception as e:
            query_classification, chosen_elaborate, rejected_elaborate = "", "", ""

        outputs.append({
            "query_classification": query_classification,
            "chosen_elaborate": chosen_elaborate,
            "rejected_elaborate": rejected_elaborate
        })

        if (idx + 1) % 50 == 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)

        if len(outputs) % 50 != 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
    app.run(main)
