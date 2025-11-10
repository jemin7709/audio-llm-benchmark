from typing import List, Dict
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


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
        )
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
