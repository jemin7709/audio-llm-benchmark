from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.utils.attention_io import select_layers


class Gemma3N:
    def __init__(
        self,
        name: str = "google/gemma-3n-E4B-it",
        dtype: str = "auto",
        device: str = "cuda:0",
        seed: int = 42,
    ):
        # tp = torch.cuda.device_count()
        # dummpy = [torch.randn((10, 3, 1024, 1024)).to(f"cuda:{i}") for i in range(tp)]
        # print(f"Dummy matrix initialized on {tp} GPUs")
        # time.sleep(3)

        # seed_everything(seed)

        self.model = AutoModelForImageTextToText.from_pretrained(
            name,
            dtype=dtype,
            device_map=device,
            attn_implementation="eager",
        )
        if hasattr(self.model, "set_attn_implementation"):
            # Ensure HF sticks to standard SDP implementation so output_attentions works.
            self.model.set_attn_implementation("eager")
        self.processor = AutoProcessor.from_pretrained(name)

        # del dummpy
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

    def _render(self, messages: List[Dict[str, str]]) -> str:
        return (
            self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            .to(self.model.device)
            .to(self.model.dtype)
        )

    def generate(self, messages) -> str:
        inputs = self._render(messages)
        with torch.no_grad():
            text_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=2048,
            )
        input_length = int(inputs["input_ids"].shape[1])
        gen_only = text_ids[:, input_length:] if input_length > 0 else text_ids
        text_out = self.processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return text_out

    def extract_attentions(
        self,
        messages: List[Dict[str, str]],
        layers: Optional[Sequence[int]] = None,
    ) -> Dict[str, List]:
        """
        Returns per-layer attention maps after averaging across heads.
        """

        if not hasattr(self.model.config, "output_attentions"):
            raise RuntimeError("This model config does not support attention outputs.")

        inputs = self._render(messages)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
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


if __name__ == "__main__":
    model = Gemma3N()
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt},
        ]
        print(model.generate(messages))
