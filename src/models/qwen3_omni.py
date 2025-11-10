from typing import List, Dict
import torch
from qwen_omni_utils import process_mm_info
from transformers.models.qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from vllm import LLM, SamplingParams
import time
import os


class Qwen3Omni:
    def __init__(
        self,
        name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        dtype: str = "auto",
        device: str = "auto",
        seed: int = 42,
        use_vllm: bool = False,
    ):
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.use_vllm = use_vllm

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if self.use_vllm:
            self.llm = LLM(
                model=name,
                trust_remote_code=True,
                gpu_memory_utilization=0.8,
                tensor_parallel_size=torch.cuda.device_count(),
                limit_mm_per_prompt={"image": 3, "video": 3, "audio": 3},
                max_model_len=4096,
                seed=seed,
            )
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=2048,
            )
        else:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                name,
                dtype=dtype,
                attn_implementation="flash_attention_2",
                max_length=4096,
                device_map=device,
            )
            self.model.disable_talker()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(name)

    def _render(self, messages: List[Dict[str, str]]) -> str:
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        if self.use_vllm:
            inputs = {
                "prompt": text_prompt,
                "multi_modal_data": {},
                "mm_processor_kwargs": {
                    "use_audio_in_video": True,
                },
            }
            if images is not None:
                inputs["multi_modal_data"]["image"] = images
            if videos is not None:
                inputs["multi_modal_data"]["video"] = videos
            if audios is not None:
                inputs["multi_modal_data"]["audio"] = audios
        else:
            inputs = (
                self.processor(
                    text=text_prompt,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=True,
                )
                .to(self.model.device)
                .to(self.model.dtype)
            )
        return inputs

    def generate(self, messages) -> str:
        inputs = self._render(messages)
        if self.use_vllm:
            outputs = self.llm.generate([inputs], sampling_params=self.sampling_params)
            return outputs[0].outputs[0].text.strip()
        else:
            with torch.no_grad():
                text_ids, _ = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=2048,
                    use_audio_in_video=True,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            input_length = int(inputs["input_ids"].shape[1])
            gen_only = (
                text_ids.sequences[:, input_length:] if input_length > 0 else text_ids
            )
            text_out = self.processor.batch_decode(
                gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            return text_out


if __name__ == "__main__":
    for use_vllm in [True, False]:
        print(f"Using vLLM: {use_vllm}")
        model = Qwen3Omni(use_vllm=use_vllm)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav",
                    },
                    {
                        "type": "text",
                        "text": "What is the content of the audio?",
                    },
                ],
            }
        ]
        print(model.generate(messages))
        print("-" * 100)
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(3)
