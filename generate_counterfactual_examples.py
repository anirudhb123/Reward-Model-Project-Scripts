r"""Generate counterfactual responses for a given dataset by modifying rejected responses into a list format.

Example usage:

INPUT_PATH=/mnt/nlpgridio3/data/anirudh2/data/GEMMA_training_sample.jsonl
LABELED_PATH=/mnt/nlpgridio3/data/anirudh2/data/GEMMA_sample_labeled_list.jsonl
OUTPUT_PATH=/mnt/nlpgridio3/data/anirudh2/data/GEMMA_counterfactuals.jsonl
MODEL_NAME=gpt-4
python3 generate_counterfactual_examples.py \
--input_path=${INPUT_PATH} \
--labeled_path=${LABELED_PATH} \
--output_path=${OUTPUT_PATH} \
--model_name=${MODEL_NAME}
"""

from absl import app, flags
import json
import tqdm
import models  # Import models for API call

# Define flags
_INPUT_PATH = flags.DEFINE_string("input_path", "", "Path to the input file.")
_LABELED_PATH = flags.DEFINE_string("labeled_path", "", "Path to the labeled list file.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "", "Path to the output file.")
_MODEL_NAME = flags.DEFINE_string("model_name", "gpt-4", "Model name for API.")

def convert_to_list(query, response, model):
    """Use API model to convert a response to a list format while preserving content."""
    prompt = f"""Adjust the original answer so it’s more structured, but change *nothing* else.

    Query: {query}

    Response: {response}

    Output:"""

    try:
        generated_response = model.generate(input_text=prompt, max_len=2048)
        return generated_response.strip()
    except Exception as e:
        print(f"Error calling model API: {e}")
        return response  # Fallback to original response if API call fails

def convert_to_text(query, response, model):
    """Use API model to convert a response to a list format while preserving content."""
    prompt = f"""Adjust the original answer so it’s less structured, but change *nothing* else.

    Query: {query}

    Response: {response}

    Output:"""

    try:
        generated_response = model.generate(input_text=prompt, max_len=2048)
        return generated_response.strip()
    except Exception as e:
        print(f"Error calling model API: {e}")
        return response  # Fallback to original response if API call fails


def main(unused_argv):
    labeled_path = _LABELED_PATH.value
    input_path = _INPUT_PATH.value
    output_path = _OUTPUT_PATH.value
    model_name = _MODEL_NAME.value

    # Load model
    model = models.GPT4(model_name=model_name) if "gpt" in model_name else models.TogetherAI(model_name=model_name)

    # Load labeled data
    labeled_data = []
    with open(labeled_path, "r") as f:
        for line in f:
            labeled_data.append(json.loads(line))

    buffer = []
    count = 0
    examples = 0
    with open(input_path, "r") as input_file, open(output_path, "a") as output_file:
        for line, label in tqdm.tqdm(zip(input_file, labeled_data), total=len(labeled_data), desc="Processing responses"):
            data = json.loads(line)

            # if label["chosen_list"] == "No" and label["rejected_list"] == "No":
            #     modified_rejected = convert_to_list(data["query"], data["rejected"], model)
            #     counterfactual_entry = {
            #         "query": data["query"],
            #         "chosen_response": data["chosen"],
            #         "rejected_response": modified_rejected
            #     }

            if label["chosen_list"] == "Yes" and label["rejected_list"] == "Yes" and examples < 500:
                modified_chosen = convert_to_text(data["query"], data["chosen"], model)
                counterfactual_entry = {
                    "query": data["query"],
                    "chosen_response": modified_chosen,
                    "rejected_response": data["rejected"],
                }
                examples += 1
            else:
                continue

            buffer.append(counterfactual_entry)
            count += 1

            if len(buffer) >= 50:
                for entry in buffer:
                    output_file.write(json.dumps(entry) + "\n")
                output_file.flush()
                buffer = []


    print(f"Counterfactuals generated and saved to {output_path}")

if __name__ == "__main__":
    app.run(main)
