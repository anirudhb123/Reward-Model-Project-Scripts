import json
import csv
import argparse
import pandas as pd
import random

def convert_jsonl_to_csv(jsonl_path, csv_path):
    questions = []
    with open(jsonl_path, 'r') as jsonl_file:
        data = [json.loads(line.strip()) for line in jsonl_file]

    code_keywords = {'code', 'script', 'function', 'method', 'class', 'program', 'python', 'java', 'javascript', 'c++', 'html', 'sql', 'c', 'ssl', 'exe', 'korea', 'grandma', 'faster', 'square-root', '89/7', 'next.js', 'fastapi', 'calculate', 'configure'}

    # data = data[:100]
    cleaned_data = []
    for d in data:
        words = d['query'][:-1].split(' ')
        if words[0].lower() in set(['who', 'what', 'when', 'where', 'why', 'how']) and not any(word.lower() in code_keywords for word in words):
            cleaned_data.append(d)
            questions.append(d['query'])

    data = [cleaned_data[i] for i in random.sample(range(len(cleaned_data)), 100)]
    print(len(cleaned_data))

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            "question", "response_1", "response_2", "reward_model_preferred_response",
            "completed", "locked", "current_date"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for record in data:
            query = record["query"]
            base_response = record["base_response"]
            perturbed_response = record["perturbed_response"]
            base_score = record["base_score"]
            perturbed_score = record["perturbed_score"]

            if hash(query) % 2 == 0:
                response_1, response_2 = base_response, perturbed_response
                response_1_score, response_2_score = base_score, perturbed_score
            else:
                response_1, response_2 = perturbed_response, base_response
                response_1_score, response_2_score = perturbed_score, base_score

            reward_model_preferred_response = (
                "response_1" if response_1_score > response_2_score else "response_2"
            )

            for _ in range(3):
                writer.writerow({
                    "question": query,
                    "response_1": response_1,
                    "response_2": response_2,
                    "reward_model_preferred_response": reward_model_preferred_response,
                    "completed": "false",
                    "locked": "false",
                    "current_date": ""
                })

        print(questions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a JSONL file to a CSV file.")
    parser.add_argument("jsonl_path", help="Path to the input JSONL file")
    parser.add_argument("csv_path", help="Path to the output CSV file")
    args = parser.parse_args()

    convert_jsonl_to_csv(args.jsonl_path, args.csv_path)

    print(f"CSV file created at: {args.csv_path}")
