from pathlib import Path

import pandas as pd
import os
import json
import re
from models import Qwen3Omni, Qwen2_5Omni  # noqa: F401
import random
from datetime import datetime, timezone

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

LLM_EXTRACTOR_SYSTEM_PROMPT = """
You are given a prompt and an audio file as inputs, along with a pair of responses to them. One of the responses is from a trained model and theother is from a baseline model.
Both responses are evaluated using an existing rubric. Your task is to identify their differences not already covered bythe existing rubrics.
You should find the properties of one response that are better than the other.
Also, try to identify reward hacking patterns in the responses.
Reward hacking is a pattern where the response achieves a high score on rubrics by exploiting a loophole in the rubrics.Think of reward hacking as a way to game the rubrics to get a high score. Reward hacking is like following the letter ofthe law but not the spirit of the law.

First, analyze both responses to identify the differences. Then, transform these observations into new evaluation criteria
if they're not already covered by existing rubrics.
This is very important, any rubric that you introduce should be based on one of the responses.
Do not use your own knowledge to introduce new criteria that are not based on one of the responses.
Focus on criteria that distinguish genuinely helpful responses from those gaming the system. Also, keep an eye out forlanguage switching patterns that might confuse the verifier.
Make sure the new criteria follow the same style as the existing criteria.
Assign a positive weight (integer) to each of the new criteria based on the relative importance of the criterion to theexisting criteria.

Output format:
```json
{
    "analysis": "Your analysis of reward hacking patterns in the responses and good/bad behaviors that should be encouraged/discouraged. It's okay for the analysis to be long.",
    "new_criteria": [
        {
            "quote": "quote from the response following/violating the criterion",
            "criterion": "criterion_text",
            "weight": criterion_weight,
        }
    ]
}
```
If no meaningful new criteria are needed, output: 
```json
{
    "analysis": "Your analysis...",
    "new_criteria": []
}
```
"""

LLM_EXTRACTOR_PROMPT = """
Prompt:
{PROMPT}

Response from the trained model:
{RESPONSE_FROM_TRAINED_MODEL}

Response from the baseline model:
{RESPONSE_FROM_BASELINE_MODEL}

Existing rubrics:
{EXISTING_RUBRICS}
"""

DE_DUPLICATE_SYSTEM_PROMPT = """
You will review a collection of candidate evaluation criteria from multiple response comparisons and remove redundancy whilepreserving the best unique criteria. Your goal is ONLY to deduplicate and aggregate, NOT to introduce new criteria or remove criteriaentirely.

## Your Task: Deduplication and Aggregation ONLY 

You should:
- **Remove redundant/overlapping criteria** that say essentially the same thing
- **Merge similar criteria** by combining them into a single, clearer criterion
- **Aggregate weights** for merged criteria (e.g., if two similar criteria have weights 3.0 and 4.0, the merged criterion might get weight 3 or 4). 
- **Preserve all unique criteria** that address different quality aspects
- **Keep the original wording** when possible, only clarifying when necessary

You should NOT:
- **Add completely new criteria** not present in the candidate list
- **Remove criteria entirely** unless they are truly redundant
- **Change the intent** of existing criteria
- **Introduce your own knowledge** beyond what's in the candidates

## Deduplication Process
1. **Group similar criteria** - Identify candidates that address the same quality aspect
2. **Select best wording** - Choose the clearest, most specific wording from each group
3. **Aggregate weights** - Combine weights from merged criteria appropriately. Only use positive integers.
4. **Preserve unique criteria** - Keep all criteria that address different aspects
5. **Maintain quality focus** - Ensure the final set covers all important quality dimensions from candidates

## CRITICAL: You MUST end your response with JSON
```json
{
    "analysis": "Your analysis of redundancy patterns and merging decisions...",
    "final_criteria": [
        {
            "criterion": "Deduplicated criterion text (merged from similar candidates)",
            "weight": criterion_weight
        },
        {
            "criterion": "Original candidate criterion 1",
            "weight": criterion_weight
        },
    ]
}
```

If all criteria are unique (no deduplication needed), return all candidates:
```json
{
    "analysis": "No redundancy found, all criteria are unique...",
    "final_criteria": [
        {
            "criterion": "Original candidate criterion 1",
            "weight": criterion_weight
        },
        {
            "criterion": "Original candidate criterion 2",
            "weight": criterion_weight
        }
    ]
}
```
"""


def load_clotho_csv(split: str) -> pd.DataFrame:
    """Clotho 데이터셋 CSV를 직접 로드합니다 (오디오 없이 캡션만)."""
    # HuggingFace 캐시 경로
    hf_cache = Path(os.environ["HF_HOME"]) / "hub"

    # 데이터셋 경로 찾기
    clotho_dirs = list(hf_cache.glob("datasets--woongvy--clotho-v2.1/snapshots/*"))
    if not clotho_dirs:
        raise FileNotFoundError(
            "Clotho 데이터셋을 찾을 수 없습니다. 먼저 데이터셋을 다운로드하세요."
        )

    # 가장 최신 스냅샷 사용
    csv_path = clotho_dirs[0] / f"clotho_captions_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    return df, clotho_dirs[0]


def extract_json_from_code_fence(text) -> dict | str:
    """```json ... ``` 또는 ``` ... ```에서 JSON을 추출해 파싱. 실패 시 '에러' 반환.

    단순화 정책:
    - 코드펜스 본문을 그대로 json.loads 시도
    - 실패 시 트레일링 콤마(,} ,])만 제거해서 한 번 더 시도
    - 그래도 실패하면 '에러' 반환
    """
    if not isinstance(text, str):
        return "에러"

    def try_parse(s: str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    # 코드펜스 우선
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate = m.group(1).strip() if m else text.strip()

    parsed = try_parse(candidate)
    if parsed is not None:
        return parsed

    # 트레일링 콤마 제거 후 재시도
    sanitized = re.sub(r",\s*(\})", r"\1", candidate)
    sanitized = re.sub(r",\s*(\])", r"\1", sanitized)
    parsed = try_parse(sanitized)
    if parsed is not None:
        return parsed

    return "에러"


def main():
    split = "evaluation"
    df, clotho_base_path = load_clotho_csv(split)

    model = Qwen3Omni()

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    rubric = []
    prompt = "Describe the sound in detail."

    for idx, row in df.iterrows():
        output_path = output_dir / f"results_{idx}.json"

        conversation1 = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "audio",
                        "audio": os.path.join(
                            clotho_base_path, split, row["file_name"]
                        ),
                    },
                ],
            }
        ]
        baseline_response = model.generate(conversation1)

        selected_caption_index = random.randint(1, 5)
        selected_caption_text = row[f"caption_{selected_caption_index}"]
        conversation2 = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": LLM_EXTRACTOR_SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": LLM_EXTRACTOR_PROMPT.format(
                            PROMPT=prompt,
                            RESPONSE_FROM_TRAINED_MODEL=selected_caption_text,
                            RESPONSE_FROM_BASELINE_MODEL=baseline_response,
                            EXISTING_RUBRICS=rubric,
                        ),
                    },
                    {
                        "type": "audio",
                        "audio": os.path.join(
                            clotho_base_path, split, row["file_name"]
                        ),
                    },
                ],
            },
        ]
        extractor_response = model.generate(conversation2)
        extractor_response_json = extract_json_from_code_fence(extractor_response)

        de_duplicate_response = ""
        de_duplicate_response_json = {}
        if isinstance(extractor_response_json, dict):
            new_criteria = [
                {
                    "criterion": criterion["criterion"],
                    "weight": criterion["weight"],
                }
                for criterion in extractor_response_json["new_criteria"]
            ]
            rubric.extend(new_criteria)
            conversation3 = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": DE_DUPLICATE_SYSTEM_PROMPT,
                        }
                    ],
                },
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(rubric, ensure_ascii=False, indent=2),
                        }
                    ],
                },
            ]
            de_duplicate_response = model.generate(conversation3)
            de_duplicate_response_json = extract_json_from_code_fence(
                de_duplicate_response
            )
            if isinstance(de_duplicate_response_json, dict):
                rubric = [
                    {
                        "criterion": criterion["criterion"],
                        "weight": criterion["weight"],
                    }
                    for criterion in de_duplicate_response_json["final_criteria"]
                ]

        record = {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "row_index": int(idx),
            "split": split,
            "audio_file": os.path.join(clotho_base_path, split, row["file_name"]),
            "file_name": row["file_name"],
            "prompt": prompt,
            "captions": [
                row.get("caption_1"),
                row.get("caption_2"),
                row.get("caption_3"),
                row.get("caption_4"),
                row.get("caption_5"),
            ],
            "selected_caption_index": int(selected_caption_index),
            "selected_caption_text": selected_caption_text,
            "baseline_response": str(baseline_response),
            "extractor_response": str(extractor_response),
            "extractor_response_json": extractor_response_json,
            "de_duplicate_response": str(de_duplicate_response),
            "de_duplicate_response_json": de_duplicate_response_json,
            "rubric": rubric,
        }
        results.append(record)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
