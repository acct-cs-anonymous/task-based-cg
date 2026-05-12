"""
Module for loading pretrained language models from HuggingFace.

This module provides utilities to download and load pretrained transformer models
like Llama, GPT-2, etc. for fine-tuning on task-based compositional generalization.
"""

import os
from typing import Optional
from init import ROOT_DIR
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


LLAMA3_MODEL_NAME = "meta-llama/Llama-3.1-8B"
GPT_OSS_MODEL_NAME = "openai/gpt-oss-20b"
GRANITE_MODEL_NAME = "ibm-granite/granite-3.1-2b-base"
GEMMA_1B_MODEL_NAME = "google/gemma-3-1b-it"
# Set environment variables to control cache locations
def set_cache_env_vars(model_name: str):
    # Set HF_HOME to control all HuggingFace cache locations
    if model_name == LLAMA3_MODEL_NAME:
        postfix = "llama3"
    elif model_name == GPT_OSS_MODEL_NAME:
        postfix = "gpt"
    elif model_name == GRANITE_MODEL_NAME:
        postfix = "ibm-granite"
    elif model_name == GEMMA_1B_MODEL_NAME:
        postfix = "gemma1"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    cache_dir = Path( str(ROOT_DIR) + f".cache/huggingface")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / "transformers")
    os.environ['HF_DATASETS_CACHE'] = str(cache_dir / "datasets")
    os.environ['TMPDIR'] = str(cache_dir / "tmp")

    print(f"Set cache environment variables to: {cache_dir}")

class PretrainedModelLoader:
    def __init__(
        self,
        model_name: str = LLAMA3_MODEL_NAME,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        **model_kwargs,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(ROOT_DIR, "cache", "models")
        self.device = device
        self.model_kwargs = model_kwargs

        set_cache_env_vars(self.model_name)

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.config = None

    def load_model_and_tokenizer(self, torch_dtype=torch.bfloat16):
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Using dtype: {torch_dtype}")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load config
        print("Loading config...")
        print(self.model_name)
        print(self.cache_dir)
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Load model
        print("Loading model...")
        model_kwargs = {
            **self.model_kwargs,
            "torch_dtype": torch_dtype,
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Move model to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(self.device)
            print(f"Model moved to {self.device}")
        elif self.device == "cuda":
            print("CUDA not available, using CPU")
            self.device = "cpu"
            self.model = self.model.to("cpu")

        # Enable gradient checkpointing to save memory (optional)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        print(f"Model loaded successfully!")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print(f"Model parameters: {self.count_parameters()}")

        return self.model, self.tokenizer, self.config

    def count_parameters(self, trainable_only: bool = False) -> int:
        if self.model is None:
            return 0

        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())

    def get_model_info(self) -> dict:
        if self.model is None:
            return {}

        return {
            "model_name": self.model_name,
            "device": self.device,
            "total_parameters": self.count_parameters(trainable_only=False),
            "trainable_parameters": self.count_parameters(trainable_only=True),
            "vocab_size": self.config.vocab_size if self.config else None,
            "max_position_embeddings": (
                self.config.max_position_embeddings if self.config else None
            ),
            "hidden_size": self.config.hidden_size if self.config else None,
            "num_layers": self.config.num_hidden_layers if self.config else None,
            "num_attention_heads": (
                self.config.num_attention_heads if self.config else None
            ),
        }

def load_pretrained_model(
    model_name: str = "meta-llama/Llama-3.1-8B",
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    torch_dtype=torch.bfloat16,
    **kwargs,
):
    
    loader = PretrainedModelLoader(
        model_name=model_name, cache_dir=cache_dir, device=device, **kwargs
    )
    model, tokenizer, config = loader.load_model_and_tokenizer(torch_dtype=torch_dtype)
    print(f"\nModel info:")
    for key, value in loader.get_model_info().items():
        print(f"  {key}: {value}")
    return model, tokenizer, config


def load_llama3_8b(device: str = "cuda", cache_dir: Optional[str] = None, torch_dtype=torch.bfloat16):
    return load_pretrained_model(
        model_name=LLAMA3_MODEL_NAME,
        cache_dir=cache_dir,
        device=device,
        torch_dtype=torch_dtype,
    )


def load_gpt_oss_20b(device: str = "cuda", cache_dir: Optional[str] = None, torch_dtype=torch.bfloat16):
    return load_pretrained_model(
        model_name=GPT_OSS_MODEL_NAME,
        cache_dir=cache_dir,
        device=device,
        torch_dtype=torch_dtype,
    )


def load_granite_2b(device: str = "cuda", cache_dir: Optional[str] = None, torch_dtype=torch.bfloat16):
    return load_pretrained_model(
        model_name=GRANITE_MODEL_NAME,
        cache_dir=cache_dir,
        device=device,
        torch_dtype=torch_dtype,
    )

def load_gemma_1b(device: str = "cuda", cache_dir: Optional[str] = None, torch_dtype=torch.bfloat16):
    return load_pretrained_model(
        model_name=GEMMA_1B_MODEL_NAME,
        cache_dir=cache_dir,
        device=device,
        torch_dtype=torch_dtype,
    )

if __name__ == "__main__":
    # Example usage
    print("Testing model loader...")
    print("=" * 50)
    model, tokenizer, config = load_granite_2b()
    # model, tokenizer, config = load_gpt_oss_20b()