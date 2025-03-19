# This script exists just to load models faster
import functools
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)

from _settings import MODEL_PATH


@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=torch.float16):
    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, "Llama-2-7b-chat-hf"), cache_dir=None, torch_dtype=torch_dtype)
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, "Llama-3.1-8B-Instruct"), cache_dir=None, torch_dtype=torch_dtype)
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, "Mistral-7B-Instruct-v0.3"), cache_dir=None, torch_dtype=torch_dtype)
    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "Llama-2-7b-chat-hf"), cache_dir=None, use_fast=use_fast)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "Llama-3.1-8B-Instruct"), cache_dir=None, use_fast=use_fast)
        # Ensure padding token is set properly
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "Mistral-7B-Instruct-v0.3"), cache_dir=None, use_fast=use_fast)
        # Ensure padding token is set properly
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token