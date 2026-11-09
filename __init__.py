"""InsideGPT: a small, modular GPT built to be opened up and inspected."""

from .config import GPTConfig, TrainConfig, resolve_device
from .tokenizer import CharTokenizer
from .hooks import ActivationCache, HookPoint
from .model import GPT
from .checkpoint import load_model, save_checkpoint
from .probe import inspect_prompt

__all__ = [
    "GPTConfig",
    "TrainConfig",
    "resolve_device",
    "CharTokenizer",
    "ActivationCache",
    "HookPoint",
    "GPT",
    "load_model",
    "save_checkpoint",
    "inspect_prompt",
]
