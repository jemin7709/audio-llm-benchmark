from .qwen2_5_omni import Qwen2_5Omni
from .qwen3_omni import Qwen3Omni
from .gemma3n import Gemma3N

REGISTRY = {
    "qwen2_5-omni": lambda **kwargs: Qwen2_5Omni("Qwen/Qwen2.5-Omni-7B"),
    "qwen3-omni": lambda **kwargs: Qwen3Omni(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", **kwargs
    ),
    "gemma3n": lambda **kwargs: Gemma3N("google/gemma-3n-E4B-it"),
}


def load_model(name: str, use_vllm: bool = False):
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[key](use_vllm=use_vllm)
