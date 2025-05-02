

r"""Analyze queries and classify responses into hedging-related categories.

Example usage:

INPUT_PATH=data/GEMMA_training_sample.jsonl
OUTPUT_PATH=data/GEMMA_sample_labeled.jsonl
python3 main/hedging_labeling.py \
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
    "output_path", "", "Path to the output file with labeled hedging-related classifications."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4", "Model name to use for classification."
)

def main(unused_argv) -> None:
    classification_prompt = """
You are a query classifier.

A query “benefits from hedging” if it meets any of these criteria:
1. **Open‑ended:** No single correct answer; invites discussion or exploration.
2. **Opinion‑based:** Asks for subjective views or covers polarizing topics.
3. **Comparative:** Requires comparing two or more options.

Hedging means using language that expresses uncertainty (e.g. “might,” “could,” “often”), mitigates claims, or acknowledges multiple perspectives without committing to one.

1. **Query benefits from hedging?**  
   Answer **Yes** or **No** only.

2. **Chosen Response Contains Hedging?**  
   Does the chosen response include hedging language as defined above? Answer **Yes** or **No**.

3. **Rejected Response Contains Hedging?**  
   Does the rejected response include hedging language as defined above? Answer **Yes** or **No**.

Here are the query and responses:

- **Query:** [QUERY]  
- **Chosen Response:** [CHOSEN]  
- **Rejected Response:** [REJECTED]  

Provide your answers in exactly this format (one per line):

Query benefits from hedging: [Yes/No]  
Chosen contains hedging: [Yes/No]  
Rejected contains hedging: [Yes/No]
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
            asked_for_hedging = parsed_lines[0] if len(parsed_lines) > 0 else ""
            chosen_hedging = parsed_lines[1] if len(parsed_lines) > 1 else ""
            rejected_hedging = parsed_lines[2] if len(parsed_lines) > 2 else ""
        except Exception as e:
            asked_for_hedging, chosen_hedging, rejected_hedging = "", "", ""

        outputs.append({
            "asked_for_hedging": asked_for_hedging,
            "chosen_hedging": chosen_hedging,
            "rejected_hedging": rejected_hedging
        })

        if (idx + 1) % 50 == 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)

        if len(outputs) % 50 != 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
    app.run(main)

