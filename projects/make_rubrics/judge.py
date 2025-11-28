import argparse
import json
from src.models import load_model

LLM_GRADER_SYSTEM_PROMPT = """
You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please evaluate the response against EACH rubric.

For each rubric,
- Mark "PRESENT" if the criterion is satisfied, or "NOT_PRESENT" if it is not. For example, given the response "Apples are red", therubric "Mentions apples" is PRESENT, "Does not mention strawberries" is also PRESENT since the response doesn't mentionstrawberries and "Mentions oranges" is NOT_PRESENT. Also, "Avoids mentioning strawberries" is PRESENT because the responsedoesn't mention strawberries. However, "Avoids mentioning apples" is NOT_PRESENT because the response mentions apples. 
- If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should beNOT PRESENT. Only return PRESENT if all of the criteria are met. 
- One important exception to the above bullet point is that if a rubric says "such as", "for example", or "including", the response does nothave to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can leadto unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral ironsupplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn'tmention any of the specific examples listed in the criteria. That is, there are no partial credit for any of the criteria. 

Start your response with a valid JSON object that starts with "```json" and ends with "```".

The keys must be the numbers of the rubrics provided and the values must be either "PRESENT" or "NOT_PRESENT" based on yourevaluation. Ensure the JSON is valid and contains no extra text or explanations. 

Example response:
```json
{
    "Criterion_1": "PRESENT",
    "Criterion_2": "NOT PRESENT",
    "Criterion_3": "PRESENT"
}
```
"""

LLM_GRADER_PROMPT = """
User prompt:
{PROMPT}

Generated response:
{GENERATED_RESPONSE}

Rubrics:
{RUBRICS}
"""


def main(args):
    with open(args.rubric_path, "r") as f:
        rubrics = json.load(f)
    with open(args.model_outputs_path, "r") as f:
        model_outputs = json.load(f)

    model = load_model(args.model, use_vllm=args.use_vllm)

    results = []
    for i, model_output in enumerate(model_outputs):
        print(f"Model output {i}:", model_output["prediction"])
        print("-" * 100)
        print(f"Rubrics {i}:", *rubrics["rubric"])
        print("-" * 100)

        prompt = LLM_GRADER_PROMPT.format(
            PROMPT="Describe the sound in detail.",
            GENERATED_RESPONSE=model_output["prediction"],
            RUBRICS=rubrics["rubric"],
        )
        messages = [
            {"role": "system", "content": LLM_GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = model.generate(messages)
        print(response)
        results.append(
            {
                "index": i,
                "prediction": model_output["prediction"],
                "response": response,
            }
        )
        with open(args.output_json_path, "w") as f:
            json.dump(results, f, indent=4)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--rubric_path", type=str, required=True)
    parser.add_argument("--model_outputs_path", type=str, required=True)
    parser.add_argument(
        "--output_json_path", default="judged_results.json", type=str, required=True
    )
    parser.add_argument("--use_vllm", action="store_true")

    args = parser.parse_args()
    main(args)
