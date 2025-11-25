import random
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Tuple, Union

from openai import OpenAI
import librosa
import outlines
from outlines.inputs import Audio
from pydantic import BaseModel, ConfigDict, conint, constr
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from qwen_omni_utils import process_mm_info

from fense.evaluator import Evaluator
import argparse
import json
import os
import torch

# Lazy initialize the FENSE evaluator and metrics
_engine_cache = {}
_fense_evaluator = None


class Qwen3OmniOutlinesWrapper(Qwen3OmniMoeForConditionalGeneration):
    """
    Wrap Qwen3OmniMoe so that .generate() returns only the text token ids tensor,
    which is what Outlines' Transformers wrapper expects.
    """

    def generate(self, *args, **kwargs):
        output = super().generate(*args, **kwargs)

        # Qwen3 Omni can return (text_output, other_outputs, ...)
        if isinstance(output, tuple):
            text_output = output[0]
        else:
            text_output = output

        # text_output is often a GenerateOutput with `.sequences`
        sequences = getattr(text_output, "sequences", None)
        if sequences is not None:
            return sequences

        # Fallback: assume it's already a tensor
        return text_output


_CLAIRA_PROMPT = """\
You are tasked with evaluating if a set of candidate captions accurately describes the same sound in a video clip as a reference set of captions. Start by assessing the accuracy and precision of how the audio characteristics are captured in the captions, scoring from 0 to 90 based on this aspect alone. After this initial assessment, you may add additional points (from 0 to 10) based on the quality of grammar and the detailed, reasonable descriptions present in the captions.

Candidate set:
{candidate_statements}

Reference set:
{target_statements}

Combine these two aspects for a final evaluation score on a scale from 0 to 100, reflecting the likelihood that the candidate set is describing the same sound as the reference set. Format your response in JSON with a key "score", value between 0 and 100, and a key "reason" with a string value explaining your assessment.
"""


class CLAIRAResponse(BaseModel):
    score: Annotated[int, conint(ge=0, le=100)]
    reason: Annotated[str, constr(max_length=1024)]


class CLAIRAReponseOpenAI(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int
    reason: str


def _find_evaluation_root() -> Path:
    base = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--woongvy--clotho-v2.1"
        / "snapshots"
    )
    if not base.exists():
        raise FileNotFoundError(f"Snapshots base not found: {base}")

    snapshot_dirs = sorted(base.iterdir())
    for snapshot_dir in snapshot_dirs:
        evaluation_dir = snapshot_dir / "evaluation"
        if evaluation_dir.exists():
            return evaluation_dir

    raise FileNotFoundError(f"No evaluation directory found under {base}")


def _engine_from_cache(
    model: str,
) -> Tuple[callable, Union[type[CLAIRAResponse], type[CLAIRAReponseOpenAI]]]:
    # Initialize the generator using outlines
    if model not in _engine_cache:
        if model.startswith("openai/"):
            # Use new outlines API for OpenAI models
            model_name = model[len("openai/") :]
            client = OpenAI()  # Uses OPENAI_API_KEY environment variable
            outlines_model = outlines.from_openai(client, model_name)
            response_type = CLAIRAReponseOpenAI
        elif model.startswith("transformers/"):
            # Use new outlines API for transformers models
            model_name = model[len("transformers/") :]
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            # hf_model = AutoModelForCausalLM.from_pretrained(
            #     model_name, device_map="auto"
            # )
            if model_name.startswith("Qwen/Qwen2.5"):
                hf_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto",
                    dtype="auto",
                    attn_implementation="flash_attention_2",
                )
                processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
                outlines_model = outlines.from_transformers(hf_model, processor)
                response_type = CLAIRAResponse
            elif model_name.startswith("Qwen/Qwen3"):
                hf_model = Qwen3OmniOutlinesWrapper.from_pretrained(
                    model_name,
                    device_map="auto",
                    dtype=torch.float32,
                    attn_implementation="sdpa",
                )
                hf_model.disable_talker()
                processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
                outlines_model = outlines.from_transformers(hf_model, processor)
                response_type = CLAIRAResponse
            elif model_name.startswith("google/gemma-3n-E4B-it"):
                hf_model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    dtype="auto",
                )
                processor = AutoProcessor.from_pretrained(model_name)
                outlines_model = outlines.from_transformers(hf_model, processor)
                response_type = CLAIRAResponse
            else:
                raise ValueError(f"Unknown model: {model_name}")
        else:
            raise ValueError(
                f"Unknown model: {model} (Prefix openai models with 'openai/', transformers models with 'transformers/')"
            )

        _engine_cache[model] = (outlines_model, response_type)
    else:
        outlines_model, response_type = _engine_cache[model]

    return outlines_model, response_type


def clair_a(
    candidate: str,
    targets: List[str],
    audio_path: str,
    model: str = "openai/gpt-4o-2024-08-06",
    tiebreaking_epsilon: float = 0.0001,
    tiebreaking_method: Union[Literal["fense"], Literal["random"]] = "fense",
) -> Tuple[float, Optional[str]]:
    # Get the outlines model
    outlines_model, response_type = _engine_from_cache(model)

    # Format the candidates and targets
    candidate_statements = [f"- {c}\n" for c in [candidate]]
    target_statements = [f"- {t}\n" for t in targets]
    formatted_prompt = _CLAIRA_PROMPT.format(
        candidate_statements="".join(candidate_statements),
        target_statements="".join(target_statements),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": formatted_prompt},
            ],
        }
    ]
    audios, _, _ = process_mm_info(messages, use_audio_in_video=True)

    prompt = [formatted_prompt, *[Audio(a) for a in audios]]
    # Use new outlines API - call model directly with output_type
    if model.startswith("openai/"):
        response_json = outlines_model(prompt, response_type)
    elif model.startswith("transformers/"):
        model_name = model[len("transformers/") :]
        if model_name.startswith("Qwen/Qwen2.5"):
            response_json = outlines_model(
                prompt,
                response_type,
                max_new_tokens=1024,
                return_audio=False,
                use_audio_in_video=True,
            )
        elif model_name.startswith("Qwen/Qwen3"):
            processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
            response_json = outlines_model(
                prompt,
                response_type,
                max_new_tokens=1024,
                return_audio=False,
                use_audio_in_video=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        elif model_name.startswith("google/gemma-3n-E4B-it"):
            # Gemma-3n expects raw audio waveform and an <audio> token in the text.
            # We load the audio using librosa and prepend the audio token to the prompt.

            # 1. Load audio (resample to 16kHz which is standard for many speech models)
            audio_waveform, sr = librosa.load(audio_path, sr=16000)
            processor = AutoProcessor.from_pretrained(model_name)
            audio_token = processor.audio_token
            gemma_prompt_text = f"{audio_token}\n{formatted_prompt}"

            # 4. Call outlines model with [text, Audio(waveform)]
            # This triggers outlines to call processor(text=..., audio=[waveform], ...)
            response_json = outlines_model(
                [gemma_prompt_text, Audio(audio_waveform)],
                response_type,
                max_new_tokens=1024,
            )
        else:
            raise ValueError(f"Unknown model: {model}")
    else:
        raise ValueError(
            f"Incompatible model: {model} (Prefix openai models with 'openai/', transformers models with 'transformers/')"
        )

    # Parse the response using Pydantic if it's a JSON string
    if isinstance(response_json, str):
        response = response_type.model_validate_json(response_json)
    elif isinstance(response_json, response_type):
        # If it's already a Pydantic model instance
        response = response_json
    else:
        raise ValueError(f"Unexpected response format: {response_json}")

    print("RESPONSE:", response, sep="\n")

    # Add the tiebreaking score
    if tiebreaking_method == "fense":
        if _engine_cache.get("_fense_evaluator") is None:
            _engine_cache["_fense_evaluator"] = Evaluator(
                device="cpu",
                sbert_model="paraphrase-mpnet-base-v2",
                echecker_model="echecker_clotho_audiocaps_tiny",
            )
        tiebreaking_score, _, _ = _engine_cache["_fense_evaluator"].sentence_score(
            candidate, targets, return_error_prob=True
        )
    elif tiebreaking_method == "random":
        tiebreaking_score = random.uniform(0, 1)

    overall_score = (response.score / 100) + tiebreaking_epsilon * tiebreaking_score

    return overall_score, response.reason


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    results = []
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        raise ValueError("CUDA_VISIBLE_DEVICES is not set")
    evaluation_root = _find_evaluation_root()
    items = json.load(open("./clotho.json"))
    for item in items:
        references = item["references"]
        for key, value in item.items():
            if key == "references" or key == "raw_name":
                continue
            for candidate in value[:2]:
                print(candidate)
                audio_path = evaluation_root / item["raw_name"]
                print(audio_path)
                score, reason = clair_a(
                    candidate,
                    references,
                    str(audio_path),
                    model="transformers/" + args.model,
                )
                print(score, reason)
                record = {
                    "raw_name": item["raw_name"],
                    "candidate": candidate,
                    "score": score,
                    "reason": reason,
                }
                results.append(record)
                json.dump(
                    results, open(f"{args.model.replace('/', '_')}.json", "w"), indent=4
                )
    json.dump(results, open(f"{args.model.replace('/', '_')}.json", "w"), indent=4)
