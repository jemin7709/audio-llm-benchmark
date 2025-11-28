from typing import Dict, List, Optional, Sequence
import torch
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

from src.utils.attention_io import select_layers


class Qwen2_5Omni:
    def __init__(
        self,
        name: str = "Qwen/Qwen2.5-Omni-7B",
        dtype: str = "auto",
        device: str = "auto",
        seed: int = 42,
    ):
        # tp = torch.cuda.device_count()
        # dummpy = [torch.randn((10, 3, 1024, 1024)).to(f"cuda:{i}") for i in range(tp)]
        # print(f"Dummy matrix initialized on {tp} GPUs")
        # time.sleep(3)

        # seed_everything(seed)

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            name,
            dtype=dtype,
            attn_implementation="eager",
            device_map=device,
        )
        if hasattr(self.model, "set_attn_implementation"):
            # Ensure HF sticks to standard SDP implementation so output_attentions works.
            self.model.set_attn_implementation("eager")
        self.processor = Qwen2_5OmniProcessor.from_pretrained(name)

        # del dummpy
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

    def _render(self, messages: List[Dict[str, str]]) -> str:
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        return (
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

    def extract_attentions(
        self, messages: List[Dict[str, str]], layers: Optional[Sequence[int]] = None
    ) -> Dict[str, List]:
        inputs = self._render(messages)
        with torch.no_grad():
            # Qwen2.5-Omni는 복합 모델이므로 thinker 서브모델을 직접 호출
            outputs = self.model.thinker(
                **inputs,
                output_attentions=True,
                use_audio_in_video=True,
                use_cache=False,
                return_dict=True,
            )

        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            raise RuntimeError("Model did not return attention tensors.")

        selected_attns, selected_layers = select_layers(attentions, layers)
        formatted_attn: List = []
        for tensor in selected_attns:
            averaged = tensor.mean(dim=1, keepdim=False)
            frame = averaged[0].to(torch.float32).detach().cpu().contiguous()
            formatted_attn.append(frame.numpy())

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor does not expose a tokenizer.")
        token_ids = inputs["input_ids"][0].detach().cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        return {
            "attentions": formatted_attn,
            "tokens": tokens,
            "layers": selected_layers,
        }

    def generate(self, messages) -> str:
        inputs = self._render(messages)
        with torch.no_grad():
            text_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=2048,
                use_audio_in_video=True,
                return_audio=False,
            )
        input_length = int(inputs["input_ids"].shape[1])
        gen_only = text_ids[:, input_length:] if input_length > 0 else text_ids
        text_out = self.processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return text_out
