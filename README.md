# Hallucination Mitigation for Large Language Models

This repository implements a comprehensive framework for evaluating and mitigating hallucinations in large language models (LLMs). The system provides a robust pipeline for detecting, categorizing, analyzing, and quantifying factual inaccuracies in model-generated content.

## Project Overview

Hallucination mitigation is a critical challenge in LLM development, addressing cases where models generate plausible-sounding but factually incorrect information. This framework offers:

- A sophisticated hallucination detection system using GPT-4o-mini as an evaluator
- Multi-dimensional analysis of hallucination severity, factual accuracy, and reliability
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

## Supported Models

The evaluation framework currently supports these models:

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

The evaluation pipeline consists of several key components:

1. **Dataset Processing**: Dataset-specific modules load and format QA pairs with appropriate prompts
2. **Model Inference**: Models generate responses based on input prompts with appropriate decoding parameters
3. **Hallucination Detection**: GPT-4o-mini evaluates responses against ground truth using a specialized prompt
4. **Statistical Analysis**: Results are aggregated and analyzed across models and datasets

### GPT-Based Evaluation

The system uses GPT-4o-mini to evaluate hallucinations through:
- A carefully designed prompt that focuses on factual accuracy and hallucination detection
- Structured JSON output with standardized scoring metrics
- Robust parsing with fallback mechanisms for handling API responses
- Detailed qualitative analysis of each hallucinated statement

### Parallel Processing

For efficiency, the system supports parallel evaluation through:
- The `TaskPartitioner` class for distributing workloads
- Configurable multi-processing for faster batch evaluation
- SLURM integration for high-performance computing environments

## Codebase Structure

```
├── data/                       # Data storage directory
│   ├── datasets/               # Raw QA datasets
│   └── weights/                # Model weights and configurations
├── dataeval/                   # Dataset processing modules
│   ├── coqa.py                 # CoQA dataset handler
│   ├── nq_open.py              # Natural Questions handler
│   ├── triviaqa.py             # TriviaQA dataset handler
│   └── SQuAD.py                # SQuAD dataset handler
├── models/                     # Model loading utilities
│   ├── _load_model.py          # Core model loading functions
│   └── __init__.py             # Model module initialization
├── utils/                      # Utility functions
│   ├── parallel.py             # Parallel processing utilities
│   └── __init__.py             # General utilities (logging, caching, etc.)
├── results/                    # Evaluation results
│   └── [MODEL_NAME]/           # Results organized by model
│       ├── result_[DATASET].txt          # Detailed per-example evaluation
│       ├── dataset_test_results_*.json   # Dataset analysis stats
│       └── hallucination_stats_*.json    # Hallucination metrics summary
├── _settings.py                # Global configuration settings
├── evaluate.py                 # Main evaluation orchestration
├── model_utils.py              # Model and dataset utilities
├── gpt_evaluation.py           # GPT-based hallucination detection
└── sbatch_evaluate.sh          # SLURM batch script for HPC environments
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

### 3. Model Management (`model_utils.py` and `models/`)
- Flexible model loading from HuggingFace or local paths
- Configurable generation parameters per dataset
- Tokenization and prompt formatting
- Abstract interfaces for consistent model interaction

### 4. Dataset Processing (`dataeval/`)
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
```

3. Set up dataset directories:
```bash
mkdir -p data/datasets data/weights
```

4. Configure API keys for evaluation:
```bash
# Add to your .bashrc or .env file
export OPENAI_API_KEY="your-api-key"
export HUGGING_FACE_HUB_TOKEN="your-hf-token"
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

#### HPC Cluster Execution

For running on SLURM-based HPC clusters:

```bash
sbatch sbatch_evaluate.sh
```

The SLURM script includes:
- Resource allocation for GPU computing
- Automatic API key loading from environment
- Multi-model evaluation configuration
- Dataset processing for all supported datasets

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

### Analyzing Results

When interpreting results, consider:

- **Hallucination Rates**: Percentage of responses with different hallucination types
- **Severity Distribution**: Distribution of hallucination severity scores
- **Dataset Differences**: How hallucination patterns vary across dataset types
- **Model Comparisons**: Relative performance of different model architectures

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
