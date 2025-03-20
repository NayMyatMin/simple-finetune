# Hallucination Mitigation for Large Language Models

This repository implements a comprehensive framework for evaluating and mitigating hallucinations in large language models (LLMs). The system provides a robust pipeline for detecting, categorizing, analyzing, and quantifying factual inaccuracies in model-generated content, as well as fine-tuning methods to reduce hallucinations.

## Project Overview

Hallucination mitigation is a critical challenge in LLM development, addressing cases where models generate plausible-sounding but factually incorrect information. This framework offers:

- A sophisticated hallucination detection system using GPT-4o-mini as an evaluator
- Multi-dimensional analysis of hallucination severity, factual accuracy, and reliability
- Parameter-efficient fine-tuning with LoRA to reduce hallucination tendencies
- Consistency training methods to improve factual reliability
- Fine-grained categorization of different hallucination types
- Support for multiple state-of-the-art models across diverse question-answering datasets
- Comprehensive statistical analysis with per-dataset and aggregate metrics

## Hallucination Detection Framework

The system employs a specialized evaluation methodology that analyzes generated responses using multiple complementary dimensions:

### Hallucination Metrics

Each evaluated response receives scores on four key dimensions:

- **Hallucination Severity** (1-10): Quantifies the amount of fabricated or unsupported information present
- **Factual Accuracy** (1-10): Measures how factually correct the answer is relative to ground truth
- **Overconfidence** (1-10): Assesses whether the model inappropriately presents speculation as fact
- **Overall Reliability** (1-10): Provides a holistic assessment of answer trustworthiness

### Hallucination Categorization

The framework classifies hallucinations into distinct categories:

- **INTRINSIC**: Content that directly contradicts the provided ground truth
- **EXTRINSIC**: Information added beyond the ground truth that cannot be verified from context
- **FACTUAL_ERROR**: Content that contains verifiably incorrect facts
- **NONE**: No hallucinations detected in the response

### Analysis Outputs

For each evaluated example, the system produces:
- Identified hallucinated statements with specific text extracts
- Detailed analysis explaining why information is considered hallucinated
- Numerical scores across all hallucination dimensions
- Per-dataset and aggregate statistics on hallucination patterns

## Fine-tuning System

The framework includes parameter-efficient fine-tuning capabilities to mitigate hallucinations using the LlamaFactory framework:

### Fine-tuning Approaches

- **Low-Rank Adaptation (LoRA)**: Parameter-efficient fine-tuning that targets specific modules in the model while keeping most parameters frozen
- **Consistency Training**: Alternative training method focused on improving factual consistency
- **DeepSpeed Integration**: Zero Optimization for efficient GPU memory usage 

### Fine-tuning Configuration

- **Target Modules**: All attention modules (q_proj, k_proj, v_proj, o_proj)
- **Training Parameters**: Configurable hyperparameters via YAML configuration files
- **Training Data**: Support for diverse instruction-tuning formats

### HPC Integration

- SLURM batch scripts for high-performance computing environments
- Distributed training support with DeepSpeed
- Gradient checkpointing and mixed-precision training for memory efficiency

## Supported Models

The framework currently supports these models:

- **Llama-2-7b-chat-hf**: Meta's Llama 2 chat model (7B parameters)
- **Llama-3.1-8B-Instruct**: Meta's latest Llama 3.1 instruction-tuned model (8B parameters)
- **Mistral-7B-Instruct-v0.3**: Mistral AI's instruction-tuned model (7B parameters)

The system is designed for easy extension to additional models by modifying the `MODEL_PATH_MAPPING` in `model_utils.py` and adding appropriate model loading code.

## Datasets

Evaluation is performed on a diverse set of question-answering datasets:

- **CoQA**: Conversational Question Answering dataset with multi-turn dialogues
- **TriviaQA**: Trivia questions requiring factual world knowledge
- **Natural Questions Open (NQ-Open)**: Questions from real Google search queries
- **SQuAD**: Stanford Question Answering Dataset with context paragraphs

Each dataset is processed through custom handlers in the `dataeval` directory, with dataset-specific prompt formatting and evaluation configurations.

## Technical Implementation

### Architecture

The project consists of two main components:

**1. Evaluation Pipeline**
- Dataset-specific modules load and format QA pairs with appropriate prompts
- Models generate responses based on input prompts with appropriate decoding parameters
- GPT-4o-mini evaluates responses against ground truth using a specialized prompt
- Results are aggregated and analyzed across models and datasets

**2. Fine-tuning Pipeline**
- LlamaFactory-based fine-tuning with LoRA for parameter-efficient adaptation
- DeepSpeed integration for memory-efficient training
- Support for both standard fine-tuning and consistency training
- YAML configuration files for flexible hyperparameter settings

### GPT-Based Evaluation

The system uses GPT-4o-mini to evaluate hallucinations through:
- A carefully designed prompt that focuses on factual accuracy and hallucination detection
- Structured JSON output with standardized scoring metrics
- Robust parsing with fallback mechanisms for handling API responses
- Detailed qualitative analysis of each hallucinated statement

### Parallel Processing

For efficiency, the system supports parallel processing through:
- The `TaskPartitioner` class for distributing evaluation workloads
- Configurable multi-processing for faster batch evaluation
- DeepSpeed for distributed training
- SLURM integration for high-performance computing environments

## Codebase Structure

```
├── configs/                    # Configuration files for fine-tuning
│   ├── consistency/            # Configuration for consistency training
│   ├── deepspeed/              # DeepSpeed optimization configurations
│   └── finetuning/             # Standard LoRA fine-tuning configs
├── data/                       # Data storage directory
│   ├── datasets/               # Raw QA datasets for evaluation
│   └── weights/                # Model weights and configurations
├── dataeval/                   # Dataset evaluation modules
│   ├── coqa.py                 # CoQA dataset handler
│   ├── nq_open.py              # Natural Questions handler
│   ├── triviaqa.py             # TriviaQA dataset handler
│   └── SQuAD.py                # SQuAD dataset handler
├── llamafactory/               # LlamaFactory integration (for fine-tuning)
│   ├── chat/                   # Chat interface components
│   ├── data/                   # Data processing for fine-tuning
│   ├── model/                  # Model loading and adaptation
│   ├── train/                  # Training loops and optimization
│   └── hparams/                # Hyperparameter management
├── lora_weight/                # Output directory for LoRA weights
├── models/                     # Model loading utilities
├── results/                    # Evaluation results
├── utils/                      # Utility functions
├── _settings.py                # Global configuration settings
├── consistency_train.py        # Consistency training entry point
├── evaluate.py                 # Main evaluation orchestration
├── finetune_train.py           # LoRA fine-tuning entry point
├── gpt_evaluation.py           # GPT-based hallucination detection
├── model_utils.py              # Model and dataset utilities
├── sbatch_consistency.sh       # SLURM script for consistency training
├── sbatch_evaluate.sh          # SLURM script for evaluation
└── sbatch_finetune.sh          # SLURM script for fine-tuning
```

## Key Components

### 1. Core Evaluation System (`evaluate.py`)
- Orchestrates the end-to-end evaluation process
- Handles dataset loading, model inference, and result collection
- Implements batch processing and progress tracking
- Creates standardized output formats across models and datasets

### 2. Hallucination Detection (`gpt_evaluation.py`)
- Implements GPT-4o-mini based detection through the OpenAI API
- Defines standardized evaluation metrics and formats
- Includes robust error handling and fallback mechanisms
- Generates both quantitative scores and qualitative analysis

### 3. Fine-tuning System (`finetune_train.py`, `consistency_train.py`)
- Implements LoRA-based fine-tuning using LlamaFactory
- Supports consistency training for improved factual reliability
- Configurable via YAML files for different models and datasets
- Integrates with DeepSpeed for efficient GPU memory usage

### 4. Model Management (`model_utils.py` and `models/`)
- Flexible model loading from HuggingFace or local paths
- Configurable generation parameters per dataset
- Tokenization and prompt formatting
- Abstract interfaces for consistent model interaction

### 5. Dataset Processing (`dataeval/`)
- Dataset-specific prompt engineering and formatting
- Custom data loading and preprocessing
- Dataset statistics and analysis
- Configurable generation settings for different data types

## Setup and Usage

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/NayMyatMin/hallucination-mitigation.git
cd hallucination-mitigation
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install torch transformers datasets tqdm requests
pip install deepspeed
```

3. Set up dataset directories:
```bash
mkdir -p data/datasets data/weights
```

4. Configure API keys for evaluation and model access:
```bash
# Add to your .bashrc or .env file
export OPENAI_API_KEY="your-api-key"
export HUGGING_FACE_HUB_TOKEN="your-hf-token"
```

### Fine-tuning Models

#### Basic Fine-tuning

To fine-tune a model using LoRA:

```bash
python finetune_train.py configs/finetuning/llama2_7b_chat/llama2_7b_finetuning_original.yaml
```

For distributed training with DeepSpeed:

```bash
torchrun --nproc_per_node=1 finetune_train.py configs/finetuning/llama2_7b_chat/llama2_7b_finetuning_original.yaml
```

#### Configuration Options

The fine-tuning configurations support various parameters in YAML format:

- Model specification (HuggingFace model ID or local path)
- Fine-tuning method (LoRA parameters and targets)
- DeepSpeed configuration for distributed training
- Training hyperparameters (learning rate, batch size, etc.)
- Dataset configuration and processing

#### HPC Execution for Fine-tuning

For running on SLURM-based HPC clusters:

```bash
sbatch sbatch_finetune.sh
```

### Running Evaluations

#### Basic Evaluation

To evaluate a specific model on all datasets:

```bash
python evaluate.py --dataset coqa triviaqa nq_open SQuAD --model Llama-2-7b-chat-hf --evaluate_with_gpt
```

#### Configuration Options

The evaluation script supports various parameters:

- `--dataset`: Specify one or more datasets to evaluate
- `--model`: Select the model to evaluate
- `--device`: Specify compute device (default: 'cuda:0')
- `--fraction_of_data_to_use`: Use a subset of data for faster testing
- `--evaluate_with_gpt`: Enable GPT-based hallucination evaluation
- `--openai_api_key`: Override OpenAI API key

#### HPC Cluster Execution for Evaluation

For running on SLURM-based HPC clusters:

```bash
sbatch sbatch_evaluate.sh
```

## Results Interpretation

### Output Files

The evaluation produces several files per model:

1. **Detailed Results**: `results/[MODEL]/result_[DATASET].txt`
   - Contains question, ground truth, model answer, and hallucination analysis
   - Includes per-example hallucination metrics and categorization
   - Provides detailed explanations of detected hallucinations

2. **Dataset Analysis**: `results/[MODEL]/dataset_test_results_[MODEL].json`
   - Contains dataset statistics (size, question length, etc.)
   - Includes tokenization analysis and potential truncation issues
   - Documents generation configuration per dataset

3. **Hallucination Statistics**: `results/[MODEL]/hallucination_stats_[MODEL].json`
   - Provides aggregated metrics across all datasets
   - Breaks down hallucination types and frequencies
   - Includes average scores for all evaluation dimensions

### Fine-tuning Outputs

The fine-tuning process produces:

1. **LoRA Weights**: `lora_weight/[MODEL]/finetuning/original/`
   - Contains adapter weights that can be loaded with the base model
   - Includes checkpoints at regular intervals during training
   - Model configuration and tokenizer files

2. **Training Logs**: Generated in the output directory or as specified in SBATCH script
   - Training progress, loss values, and metrics
   - Performance statistics and timing information
   - Hardware utilization information

### Analyzing Results

When interpreting results, consider:

- **Hallucination Rates**: Percentage of responses with different hallucination types
- **Severity Distribution**: Distribution of hallucination severity scores
- **Dataset Differences**: How hallucination patterns vary across dataset types
- **Model Comparisons**: Relative performance of different model architectures
- **Fine-tuning Impact**: Changes in hallucination metrics before and after fine-tuning

## Example Analysis Output

```
Example 42
Question: When was the first programmable electronic computer invented?
Ground Truth: The first programmable electronic computer was the ENIAC, completed in 1946.
Generated Answer: The first programmable electronic computer was invented in 1936 by Konrad Zuse, who created the Z1 computer.

GPT-4o-mini Hallucination Evaluation:
Hallucination Severity: 8.5/10
Factual Accuracy: 2.0/10
Overconfidence: 9.0/10
Overall Reliability: 1.5/10
Hallucination Type: FACTUAL_ERROR
Hallucination Examples:
  1. The first programmable electronic computer was invented in 1936
  2. It was created by Konrad Zuse
  3. Zuse created the Z1 computer
Analysis: The response contains multiple factual errors. The Z1, built by Konrad Zuse in 1936-1938, was mechanical, not electronic. According to the ground truth, the first programmable electronic computer was ENIAC, completed in 1946. The response provides incorrect information about both the date (1936 vs. 1946) and the inventor/machine (Zuse's Z1 vs. ENIAC).
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:
```
@software{nay2023hallucination,
  author = {Nay, Myat Min},
  title = {Hallucination Mitigation for Large Language Models},
  year = {2025},
  url = {https://github.com/NayMyatMin/hallucination-mitigation}
}
``` 
