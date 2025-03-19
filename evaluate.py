import argparse
import os
import torch
import tqdm
import transformers
import json
import requests
from datasets import Dataset

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import models
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, nargs='+', default=['coqa'], 
                    choices=['coqa', 'triviaqa', 'nq_open', 'SQuAD'],
                    help='One or more datasets to evaluate on')
parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf', 
                    choices=['Llama-2-7b-chat-hf', 'Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_file', type=str, default=None, help='Custom output file path (if not using the default structure)')
parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API key for GPT-4o-mini evaluation')
parser.add_argument('--evaluate_with_gpt', action='store_true', help='Enable evaluation with GPT-4o-mini')

args = parser.parse_args()

# Mapping from simplified model names to full HuggingFace paths
MODEL_PATH_MAPPING = {
    'Llama-2-7b-chat-hf': 'meta-llama/Llama-2-7b-chat-hf',
    'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3'
}

# Get model name from args and map to full path when needed
SHORT_MODEL_NAME = args.model
MODEL_NAME = MODEL_PATH_MAPPING[SHORT_MODEL_NAME]

# Set up results directory structure
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Get model folder name - use the short name directly
model_folder_name = SHORT_MODEL_NAME
model_results_dir = os.path.join(RESULTS_DIR, model_folder_name)
os.makedirs(model_results_dir, exist_ok=True)

# Check if we're doing GPT evaluation
if args.evaluate_with_gpt and not args.openai_api_key:
    print("Using OPENAI_API_KEY environment variable.")
    
# For direct HuggingFace loading - no local path used
USE_HF_DIRECTLY = True
# Set to False and use SHORT_MODEL_NAME if you have the model locally

def get_dataset_fn(data_name):
    """Return the appropriate dataset loading function"""
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'coqa':
        return coqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset
    if data_name == 'SQuAD':
        return SQuAD.get_dataset
    raise ValueError(f"Unsupported dataset: {data_name}")


def get_generation_config(input_ids, tokenizer, data_name):
    """Configure generation parameters based on dataset type"""
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    elif data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    elif data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    elif data_name == 'SQuAD':
        generation_config = SQuAD._generate_config(tokenizer)
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = False
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config


def evaluate_with_gpt4o_mini(question, ground_truth, generated_answer, api_key=None):
    """
    Evaluate the generated answer against ground truth using GPT-4o-mini
    Returns a dict with scores and explanation
    """
    if not api_key:
        # Try to get from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return {
                "error": "No OpenAI API key provided",
                "scores": {"relevance": 0, "accuracy": 0, "completeness": 0, "overall": 0}
            }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # More explicit prompt with exact JSON structure expected
    prompt = f"""You are an expert evaluator for question answering systems. Please evaluate the following:

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Rate the generated answer on the following criteria on a scale of 1-10:
1. Relevance: How relevant is the answer to the question?
2. Accuracy: How factually accurate is the answer compared to the ground truth?
3. Completeness: How complete is the answer?
4. Overall Score: An overall assessment of the answer quality.

IMPORTANT: You MUST respond with a valid JSON object that follows EXACTLY this format:
{{
  "scores": {{
    "relevance": <number between 1-10>,
    "accuracy": <number between 1-10>,
    "completeness": <number between 1-10>,
    "overall": <number between 1-10>
  }},
  "explanation": "<your brief explanation here>"
}}

Do not include any text outside of this JSON structure. The scores must be numbers, not strings.
"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an assistant that evaluates question answering systems. You MUST return your evaluation in valid JSON format."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        # Debug: print raw response
        print(f"\nAPI Response Status: {response.status_code}")
        result = response.json()
        raw_content = result["choices"][0]["message"]["content"]
        print(f"Raw API Response Content:\n{raw_content}\n")
        
        # More robust parsing with fallbacks
        try:
            # First attempt: direct JSON parsing
            evaluation = json.loads(raw_content)
            
            # Verify expected structure exists
            if "scores" not in evaluation:
                evaluation["scores"] = {}
            for metric in ["relevance", "accuracy", "completeness", "overall"]:
                if metric not in evaluation["scores"]:
                    evaluation["scores"][metric] = 0
                # Ensure scores are numeric
                try:
                    evaluation["scores"][metric] = float(evaluation["scores"][metric])
                except (ValueError, TypeError):
                    evaluation["scores"][metric] = 0
            
            if "explanation" not in evaluation:
                evaluation["explanation"] = "No explanation provided."
                
            return evaluation
        
        except json.JSONDecodeError:
            # Fallback: try to extract scores using regex
            print("JSON parsing failed, attempting regex extraction")
            import re
            
            # Extract scores using regex
            scores = {}
            for metric in ["relevance", "accuracy", "completeness", "overall"]:
                pattern = rf'"{metric}":\s*(\d+)'
                match = re.search(pattern, raw_content, re.IGNORECASE)
                scores[metric] = int(match.group(1)) if match else 0
            
            # Extract explanation
            explanation_match = re.search(r'"explanation":\s*"([^"]+)"', raw_content)
            explanation = explanation_match.group(1) if explanation_match else "No explanation extracted."
            
            return {
                "scores": scores,
                "explanation": explanation
            }
            
    except Exception as e:
        print(f"Error in API call: {e}")
        return {
            "error": str(e),
            "scores": {"relevance": 0, "accuracy": 0, "completeness": 0, "overall": 0}
        }


@torch.no_grad()
def evaluate_dataset(dataset_name, model, tokenizer, device):
    """Evaluate a single dataset using the provided model and tokenizer"""
    print(f"\n{'='*50}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*50}\n")
    
    # Set up output file for this dataset
    output_file = None
    if args.output_file:
        # User provided custom output path - append dataset name
        output_path = f"{args.output_file}_{dataset_name}"
    else:
        # Use standard format: results/MODEL_NAME/result_DATASET.txt
        output_path = os.path.join(model_results_dir, f"result_{dataset_name}.txt")
    
    output_file = open(output_path, "w", encoding="utf-8")
    print(f"Results will be saved to: {output_path}")
    
    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    dataset_fn = get_dataset_fn(dataset_name)
    dataset = dataset_fn(tokenizer)
    
    # Use only a fraction of the dataset if specified
    if args.fraction_of_data_to_use < 1.0:
        print(f"Using {args.fraction_of_data_to_use * 100}% of the dataset...")
        dataset = dataset.train_test_split(
            test_size=(1 - args.fraction_of_data_to_use), 
            seed=args.seed
        )['train']
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Aggregate evaluation metrics if using GPT
    if args.evaluate_with_gpt:
        gpt_evals = {
            "relevance": [],
            "accuracy": [],
            "completeness": [],
            "overall": []
        }
    
    # Generate answers
    print("Generating answers...")
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # Move input to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        input_length = input_ids.shape[1]
        
        # Get generation config
        generation_config = get_generation_config(input_ids, tokenizer, dataset_name)
        generation_config = transformers.GenerationConfig(**generation_config)
        
        # Generate answer with greedy decoding
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            do_sample=False,
            generation_config=generation_config,
            return_dict_in_generate=True
        )
        
        # Decode generated answer
        generated_ids = outputs.sequences[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Decode prompt
        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Get ground truth answer based on dataset format
        ground_truth = None
        if dataset_name == 'SQuAD' and 'answers' in batch:
            # SQuAD dataset has answers in a different format
            try:
                # Check if the answers field is empty first
                if len(batch['answers']) > 0:
                    answers = batch['answers'][0]
                    # Print debug information about the answers format
                    print(f"SQuAD answers format: {type(answers)}")
                    print(f"SQuAD answers content: {answers}")
                    
                    # Handle the actual SQuAD format: a dict with 'text' key containing a list of strings
                    if isinstance(answers, dict) and 'text' in answers:
                        if isinstance(answers['text'], (list, tuple)) and len(answers['text']) > 0:
                            ground_truth = answers['text'][0]  # Take the first answer text
                        elif isinstance(answers['text'], str):
                            ground_truth = answers['text']  # Direct string
                    else:
                        print("Warning: SQuAD answers format is unexpected, couldn't extract ground truth")
                else:
                    print("Warning: SQuAD example has empty answers field")
            except Exception as e:
                print(f"Error extracting SQuAD answer: {e}")
        elif 'answer' in batch:
            ground_truth = batch['answer'][0]
        
        if ground_truth is not None:
            print(f"Ground truth extracted: {ground_truth}")
        else:
            print("Ground truth extracted: None")
        
        # GPT-4o-mini evaluation
        gpt_evaluation = None
        if args.evaluate_with_gpt and ground_truth is not None:
            print(f"Evaluating example {batch_idx+1} with GPT-4o-mini...")
            gpt_evaluation = evaluate_with_gpt4o_mini(
                batch['question'][0], 
                ground_truth, 
                generated_text,
                api_key=args.openai_api_key
            )
            
            # Add to aggregate metrics
            if "scores" in gpt_evaluation and "error" not in gpt_evaluation:
                valid_scores = True
                for metric in ["relevance", "accuracy", "completeness", "overall"]:
                    if metric not in gpt_evaluation["scores"] or gpt_evaluation["scores"][metric] == 0:
                        valid_scores = False
                        print(f"Warning: Missing or zero value for {metric} score")
                
                if valid_scores:
                    for metric in ["relevance", "accuracy", "completeness", "overall"]:
                        gpt_evals[metric].append(gpt_evaluation["scores"][metric])
                else:
                    print("Warning: Some scores are missing or invalid. Not including in aggregates.")
        
        # Print results
        print("\n" + "="*80)
        print(f"Example {batch_idx+1}")
        print("-"*80)
        print(f"Question: {batch['question'][0]}")
        if ground_truth:
            print(f"Ground Truth Answer: {ground_truth}")
        print(f"Generated Answer: {generated_text}")
        
        if gpt_evaluation:
            print("-"*80)
            print("GPT-4o-mini Evaluation:")
            if "error" in gpt_evaluation:
                print(f"Error: {gpt_evaluation['error']}")
            else:
                scores = gpt_evaluation.get("scores", {})
                # Format scores with decimal places and handle missing values better
                for metric in ["relevance", "accuracy", "completeness", "overall"]:
                    score_value = scores.get(metric, 0)
                    if score_value:
                        print(f"{metric.capitalize()}: {score_value:.1f}/10")
                    else:
                        print(f"{metric.capitalize()}: Missing")
                
                if "explanation" in gpt_evaluation:
                    print(f"Explanation: {gpt_evaluation['explanation']}")
                else:
                    print("No explanation provided")
        
        print("="*80)
        
        # Write to output file
        output_file.write(f"Example {batch_idx+1}\n")
        output_file.write(f"Question: {batch['question'][0]}\n")
        if ground_truth:
            output_file.write(f"Ground Truth Answer: {ground_truth}\n")
        output_file.write(f"Generated Answer: {generated_text}\n")
        
        if gpt_evaluation:
            output_file.write("\nGPT-4o-mini Evaluation:\n")
            if "error" in gpt_evaluation:
                output_file.write(f"Error: {gpt_evaluation['error']}\n")
            else:
                scores = gpt_evaluation.get("scores", {})
                output_file.write(f"Relevance: {scores.get('relevance', 'N/A')}/10\n")
                output_file.write(f"Accuracy: {scores.get('accuracy', 'N/A')}/10\n")
                output_file.write(f"Completeness: {scores.get('completeness', 'N/A')}/10\n")
                output_file.write(f"Overall: {scores.get('overall', 'N/A')}/10\n")
                if "explanation" in gpt_evaluation:
                    output_file.write(f"Explanation: {gpt_evaluation['explanation']}\n")
        
        output_file.write("\n\n")
        
        # Free up CUDA memory
        torch.cuda.empty_cache()
    
    # Print aggregate metrics if we did GPT evaluation
    if args.evaluate_with_gpt:
        # Check if we have valid data for all metrics
        has_valid_data = all(len(scores) > 0 for scores in gpt_evals.values())
        
        if has_valid_data:
            avg_metrics = {metric: sum(scores)/len(scores) for metric, scores in gpt_evals.items()}
            
            print("\n" + "="*80)
            print("Overall GPT-4o-mini Evaluation:")
            print(f"Total evaluated examples: {len(gpt_evals['overall'])}")
            print(f"Average Relevance: {avg_metrics['relevance']:.2f}/10")
            print(f"Average Accuracy: {avg_metrics['accuracy']:.2f}/10")
            print(f"Average Completeness: {avg_metrics['completeness']:.2f}/10")
            print(f"Average Overall Score: {avg_metrics['overall']:.2f}/10")
            print("="*80)
            
            output_file.write("\n" + "="*50 + "\n")
            output_file.write("Overall GPT-4o-mini Evaluation:\n")
            output_file.write(f"Total evaluated examples: {len(gpt_evals['overall'])}\n")
            output_file.write(f"Average Relevance: {avg_metrics['relevance']:.2f}/10\n")
            output_file.write(f"Average Accuracy: {avg_metrics['accuracy']:.2f}/10\n")
            output_file.write(f"Average Completeness: {avg_metrics['completeness']:.2f}/10\n")
            output_file.write(f"Average Overall Score: {avg_metrics['overall']:.2f}/10\n")
            output_file.write("="*50 + "\n")
        else:
            print("\n" + "="*80)
            print("No valid GPT-4o-mini evaluation data collected for aggregation.")
            print("="*80)
            
            output_file.write("\n" + "="*50 + "\n")
            output_file.write("No valid GPT-4o-mini evaluation data collected for aggregation.\n")
            output_file.write("="*50 + "\n")
    
    # Close output file
    output_file.close()
    print(f"Results written to {output_path}")
    
    return output_path


@torch.no_grad()
def evaluate_model():
    """Main function to load model and evaluate on all specified datasets"""
    device = args.device
    
    # Set random seed for reproducibility
    utils.seed_everything(args.seed)
    
    # Load model and tokenizer (only once for all datasets)
    print(f"Loading model {SHORT_MODEL_NAME}...")
    
    if USE_HF_DIRECTLY:
        # Load directly from HuggingFace (requires auth)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        try:
            print(f"Attempting to load {MODEL_NAME} directly from HuggingFace Hub...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map=device
            )
            print(f"Successfully loaded {SHORT_MODEL_NAME} from HuggingFace Hub")
        except Exception as e:
            print(f"Error loading from HuggingFace Hub: {e}")
            print(f"Falling back to local model loading with name: {SHORT_MODEL_NAME}")
            model, tokenizer = models.load_model_and_tokenizer(SHORT_MODEL_NAME, device)
    else:
        # Use the regular loading method from models module with the local name
        print(f"Loading model from local path using name: {SHORT_MODEL_NAME}")
        model, tokenizer = models.load_model_and_tokenizer(SHORT_MODEL_NAME, device)
    
    # Store all output files
    output_files = []
    
    # Evaluate on each dataset
    for dataset_name in args.dataset:
        output_path = evaluate_dataset(dataset_name, model, tokenizer, device)
        output_files.append(output_path)
    
    # Print summary of all datasets evaluated
    print("\n" + "="*80)
    print(f"Evaluation complete for {SHORT_MODEL_NAME} on {len(args.dataset)} datasets:")
    for i, dataset_name in enumerate(args.dataset):
        print(f"  {i+1}. {dataset_name} - Results saved to: {output_files[i]}")
    print("="*80)


if __name__ == "__main__":
    evaluate_model()
