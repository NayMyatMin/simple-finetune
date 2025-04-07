# Simple-Finetune

A lightweight framework for parameter-efficient fine-tuning of large language models (LLMs) using LoRA (Low-Rank Adaptation). This project provides a minimal setup for fine-tuning models like Llama-2 and others.

## Overview

This framework offers a streamlined approach to fine-tuning LLMs with a focus on:

- Parameter-efficient fine-tuning with LoRA
- Support for Hugging Face models
- DeepSpeed integration for memory efficiency
- SLURM batch scripts for HPC environments
- Simple configuration via YAML files

## Features

- **LoRA Fine-tuning**: Fine-tune large models with minimal memory requirements by only updating a small set of parameters
- **DeepSpeed Integration**: Optimize memory usage and training speed
- **HPC Support**: Ready-to-use SLURM batch scripts for high-performance computing environments
- **Flexible Configuration**: Easily configure models, datasets, and training parameters via YAML files

## Supported Models

The framework currently supports:

- **Llama-2-7b-chat-hf**: Meta's Llama 2 chat model (7B parameters)
- **Llama-3.1-8B-Instruct**: Meta's latest Llama 3.1 instruction-tuned model (8B parameters)
- **Mistral-7B-Instruct-v0.3**: Mistral AI's instruction-tuned model (7B parameters)

## Setup

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/naymyatmin/simple-finetune.git
cd simple-finetune
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install torch transformers datasets 
pip install deepspeed
pip install git+https://github.com/hiyouga/LLaMA-Factory.git
```

3. Configure API keys for model access:
```bash
# Add to your .bashrc or .env file
export HUGGING_FACE_HUB_TOKEN="your-hf-token"
```

## Usage

### Fine-tuning a Model

To fine-tune a model using LoRA:

```bash
python finetune_train.py configs/finetuning/llama2_7b_chat/llama2_7b_finetuning.yaml
```

For distributed training with DeepSpeed:

```bash
torchrun --nproc_per_node=1 finetune_train.py configs/finetuning/llama2_7b_chat/llama2_7b_finetuning.yaml
```

### HPC Execution

For running on SLURM-based HPC clusters:

```bash
sbatch sbatch_finetune.sh
```

## Configuration

The fine-tuning configurations support various parameters in YAML format:

```yaml
### model
model_name_or_path: meta-llama/Llama-2-7b-chat-hf

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
deepspeed: configs/deepspeed/ds_z0_config.json

### dataset
dataset: alpaca  # or consistency, instruction, etc.
template: llama2
cutoff_len: 1024
max_samples: 1000

### output 
output_dir: lora_weight/LLaMA2-7B-Chat/finetuning/

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.001
num_train_epochs: 5.0
```

## Project Structure

```
simple-finetune/
├── configs/                    # Configuration files for fine-tuning
│   ├── deepspeed/              # DeepSpeed optimization configurations
│   └── finetuning/             # Standard LoRA fine-tuning configs
├── lora_weight/                # Output directory for LoRA weights
├── finetune_train.py           # Main entry point for fine-tuning
└── sbatch_finetune.sh          # SLURM script for fine-tuning
```

## Acknowledgments

This project is built on top of LlamaFactory and integrates with Hugging Face Transformers and DeepSpeed.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 
