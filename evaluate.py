import argparse
import os
import torch
import tqdm
import transformers
import json
from datasets import Dataset
import multiprocessing as mp
from peft import PeftModel, PeftConfig

import _settings
import model_utils
import gpt_evaluation
import utils
from utils.parallel import TaskPartitioner


def collate_fn_with_padding(batch):
    """
    Custom collate function that handles variable-length sequences by padding.
    This resolves the "stack expects each tensor to be equal size" error.
    """
    # Extract all keys from the batch
    keys = batch[0].keys()
    
    # Handle each key separately
    result = {}
    for key in keys:
        if key in ['input_ids', 'attention_mask']:
            # Handle tensors that need padding
            elements = [item[key] for item in batch]
            
            # Find max length for padding
            max_len = max(len(x) for x in elements)
            
            # Pad sequences to max length
            padded_elements = []
            for tensor in elements:
                if len(tensor) < max_len:
                    # Create a new tensor with zeros and copy the original content
                    padded = torch.zeros(max_len, dtype=tensor.dtype)
                    padded[:len(tensor)] = tensor
                    padded_elements.append(padded)
                else:
                    padded_elements.append(tensor)
            
            # Stack the padded tensors
            result[key] = torch.stack(padded_elements)
        else:
            # For non-tensor data, just use a list
            result[key] = [item[key] for item in batch]
    
    return result


def parse_arguments():
    """Parse command line arguments"""
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
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--parallel_datasets', action='store_true', help='Process datasets in parallel')
    parser.add_argument('--num_processes', type=int, default=0, 
                        help='Number of processes for parallel execution (0 for auto-detect)')
    parser.add_argument('--lora_weights', type=str, default=None,
                        help='Path to the LoRA weights directory (if not provided, uses base model)')

    return parser.parse_args()


def load_model_with_lora(base_model_name, lora_weights_path, device):
    """
    Load a base model and apply LoRA weights to it
    
    Args:
        base_model_name (str): Name of the base model
        lora_weights_path (str): Path to the LoRA weights directory
        device (str): Device to load the model on
        
    Returns:
        model, tokenizer: The model with LoRA weights applied and its tokenizer
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load the base model and tokenizer
    base_model, tokenizer = model_utils.load_model(base_model_name, device)
    
    # If no LoRA weights specified, return the base model
    if not lora_weights_path:
        print("No LoRA weights specified, using base model")
        return base_model, tokenizer
    
    # Check if the LoRA weights exist
    if not os.path.exists(lora_weights_path):
        print(f"Warning: LoRA weights path does not exist: {lora_weights_path}")
        print("Falling back to base model")
        return base_model, tokenizer
    
    print(f"Loading LoRA adapter from: {lora_weights_path}")
    
    # Load the PEFT configuration
    try:
        # Load configuration
        peft_config = PeftConfig.from_pretrained(lora_weights_path)
        print(f"LoRA config loaded: {peft_config}")
        
        # Apply the LoRA adapter
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
        print("LoRA adapter applied successfully")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        print("Falling back to base model")
        return base_model, tokenizer


def analyze_dataset(dataset, tokenizer, dataset_name):
    """Analyze dataset properties and generate statistics"""
    print(f"\nAnalyzing dataset: {dataset_name}")
    
    # Basic dataset info
    dataset_info = {
        "size": len(dataset),
        "columns": dataset.column_names,
        "column_analysis": {},
        "tokenization": {}
    }
    
    # Analyze each column
    for col in dataset.column_names:
        # Skip large tensor columns
        if col in ['input_ids', 'attention_mask']:
            continue
            
        if len(dataset) > 0 and col in dataset[0]:
            # Get sample data
            sample = dataset[0][col]
            
            # Analyze based on type
            if isinstance(sample, str):
                avg_length = sum(len(dataset[i][col]) for i in range(min(100, len(dataset)))) / min(100, len(dataset))
                dataset_info["column_analysis"][col] = {
                    "type": "str",
                    "avg_length": f"{avg_length:.2f} chars"
                }
            elif isinstance(sample, list) or isinstance(sample, tuple):
                avg_elements = sum(len(dataset[i][col]) for i in range(min(100, len(dataset)))) / min(100, len(dataset))
                dataset_info["column_analysis"][col] = {
                    "type": "list",
                    "avg_length": f"{avg_elements:.2f} elements"
                }
            elif hasattr(sample, 'shape') and hasattr(sample, 'dtype'):  # Tensor-like
                avg_elements = sum(len(dataset[i][col]) for i in range(min(100, len(dataset)))) / min(100, len(dataset))
                dataset_info["column_analysis"][col] = {
                    "type": "Tensor",
                    "avg_length": f"{avg_elements:.2f} elements"
                }
    
    # Analyze prompts if available
    if 'prompt' in dataset.column_names:
        lengths = [len(dataset[i]['prompt']) for i in range(min(100, len(dataset)))]
        dataset_info["prompt_stats"] = {
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }
    
    # Analyze tokenization
    if 'input_ids' in dataset.column_names:
        token_lengths = [len(dataset[i]['input_ids']) for i in range(min(100, len(dataset)))]
        dataset_info["tokenization"] = {
            "avg_length": sum(token_lengths) / len(token_lengths),
            "min_length": min(token_lengths),
            "max_length": max(token_lengths),
            "potential_truncation_percent": sum(1 for l in token_lengths if l > 2048) / len(token_lengths) * 100
        }
    
    # Get generation config for this dataset type
    if len(dataset) > 0 and 'input_ids' in dataset[0]:
        input_ids = dataset[0]['input_ids'].unsqueeze(0)
        generation_config = model_utils.get_generation_config(input_ids, tokenizer, dataset_name)
        dataset_info["generation_config"] = generation_config
    
    # Print and return analysis
    print(f"Dataset size: {dataset_info['size']} examples")
    if "tokenization" in dataset_info:
        print(f"Average tokens per example: {dataset_info['tokenization']['avg_length']:.1f}")
    
    return dataset_info


@torch.no_grad()
def evaluate_dataset(dataset_name, model, tokenizer, device, args, hallucination_stats=None, model_results_dir=None):
    """Evaluate a single dataset using the provided model and tokenizer"""
    print(f"\n{'='*50}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*50}\n")
    
    # Set up output file for this dataset
    output_path = model_utils.get_output_path(model_results_dir, dataset_name, args.output_file)
    output_file = open(output_path, "w", encoding="utf-8")
    print(f"Results will be saved to: {output_path}")
    
    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    dataset_fn = model_utils.get_dataset_fn(dataset_name)
    dataset = dataset_fn(tokenizer)
    
    # Use only a fraction of the dataset if specified
    if args.fraction_of_data_to_use < 1.0:
        print(f"Using {args.fraction_of_data_to_use * 100}% of the dataset...")
        dataset = dataset.train_test_split(
            test_size=(1 - args.fraction_of_data_to_use), 
            seed=args.seed
        )['train']
    
    # Analyze dataset
    dataset_info = analyze_dataset(dataset, tokenizer, dataset_name)
    
    # If running in parallel mode, reduce the number of workers to avoid tokenizer issues
    if args.parallel_datasets:
        num_workers_to_use = min(2, args.num_workers)
        print(f"Using {num_workers_to_use} workers for data loading in parallel mode")
    else:
        num_workers_to_use = args.num_workers
    
    # Set environment variable to avoid tokenizer warnings with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Create dataloader with increased batch size and parallel workers
    # Use custom collate function to handle variable-length sequences
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers_to_use,
        pin_memory=True,
        collate_fn=collate_fn_with_padding
    )
    
    # Initialize dataset-specific hallucination stats
    if hallucination_stats is not None and args.evaluate_with_gpt:
        dataset_hall_stats = {
            "total_evaluated": 0,
            "hallucination_counts": {
                "INTRINSIC": 0,
                "EXTRINSIC": 0,
                "FACTUAL_ERROR": 0,
                "NONE": 0,
                "UNKNOWN": 0
            },
            "avg_hallucination_severity": 0,
            "avg_factual_accuracy": 0,
            "avg_overconfidence": 0,
            "avg_overall_reliability": 0
        }
        hallucination_stats["dataset_stats"][dataset_name] = dataset_hall_stats
    
    # Aggregate evaluation metrics if using GPT
    if args.evaluate_with_gpt:
        gpt_evals = {
            "hallucination_severity": [],
            "factual_accuracy": [],
            "overconfidence": [],
            "overall_reliability": []
        }
    
    # Process all examples
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        # Move input to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Get generation config for this dataset
        generation_config = model_utils.get_generation_config(input_ids, tokenizer, dataset_name)
        
        # Generate response for the entire batch
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        
        # Process model outputs first
        batch_size = input_ids.size(0)
        questions = []
        ground_truths = []
        generated_answers = []
        
        for i in range(batch_size):
            # Extract the output for this example
            full_output_tokens = outputs[i]
            prompt_length = len(input_ids[i])
            generated_tokens = full_output_tokens[prompt_length:]
            
            generated_answer = tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            ).strip()
            
            # Extract original question and ground truth
            raw_question = batch['question'][i]
            
            # Create a single-example batch for extract_ground_truth
            single_batch = {k: [v[i]] if isinstance(v, list) else {k2: [v2[i]] for k2, v2 in v.items()} 
                           for k, v in batch.items() if k != 'input_ids' and k != 'attention_mask'}
            
            ground_truth = model_utils.extract_ground_truth(single_batch, dataset_name)
            
            # Store for later evaluation
            questions.append(raw_question)
            ground_truths.append(ground_truth)
            generated_answers.append(generated_answer)
        
        # Evaluate with GPT in parallel if enabled
        if args.evaluate_with_gpt:
            # Get number of API processes to use - recommend to keep this low to avoid API rate limits
            api_processes = min(4, args.num_processes) if args.num_processes > 0 else 4
            
            # Run parallel evaluation on the batch
            print(f"Evaluating batch {batch_idx+1}/{len(dataloader)} with GPT (parallel)")
            gpt_evaluation_results = gpt_evaluation.parallel_evaluate_batch(
                questions,
                ground_truths,
                generated_answers,
                api_key=args.openai_api_key,
                num_processes=api_processes
            )
            
            # Process evaluation results for each example
            for i in range(batch_size):
                # Get global index for this example
                global_idx = batch_idx * args.batch_size + i
                
                # Get question, ground truth, answer and evaluation for this example
                question = questions[i]
                ground_truth = ground_truths[i]
                generated_answer = generated_answers[i]
                gpt_evaluation_result = gpt_evaluation_results[i]
                
                # Print results 
                if global_idx < 5 or global_idx % 20 == 0:
                    print("\n" + "-"*80)
                    print(f"Example {global_idx+1}")
                    print(f"Question: {question}")
                    print(f"Ground Truth: {ground_truth}")
                    print(f"Generated Answer: {generated_answer}")
                    gpt_evaluation.print_gpt_evaluation_results(global_idx, gpt_evaluation_result)
                
                # Write to output file
                output_file.write(f"Example {global_idx+1}\n")
                output_file.write(f"Question: {question}\n")
                output_file.write(f"Ground Truth: {ground_truth}\n")
                output_file.write(f"Generated Answer: {generated_answer}\n")
                gpt_evaluation.write_gpt_evaluation_to_file(output_file, gpt_evaluation_result)
                
                # Collect scores for aggregate metrics
                if "scores" in gpt_evaluation_result:
                    scores = gpt_evaluation_result["scores"]
                    for metric in gpt_evals.keys():
                        if metric in scores and scores[metric] > 0:
                            gpt_evals[metric].append(scores[metric])
                    
                    # Update hallucination statistics
                    if hallucination_stats is not None:
                        # Update dataset stats
                        dataset_stats = hallucination_stats["dataset_stats"][dataset_name]
                        dataset_stats["total_evaluated"] += 1
                        
                        # Update hallucination type counts
                        h_type = gpt_evaluation_result.get("hallucination_type", "UNKNOWN")
                        if h_type in dataset_stats["hallucination_counts"]:
                            dataset_stats["hallucination_counts"][h_type] += 1
                        else:
                            dataset_stats["hallucination_counts"]["UNKNOWN"] += 1
                        
                        # Update overall stats
                        hallucination_stats["overall"]["total_evaluated"] += 1
                        if h_type in hallucination_stats["overall"]["hallucination_counts"]:
                            hallucination_stats["overall"]["hallucination_counts"][h_type] += 1
                        else:
                            hallucination_stats["overall"]["hallucination_counts"]["UNKNOWN"] += 1
                        
                        # Update running averages for metrics
                        for metric, key in [
                            ("hallucination_severity", "avg_hallucination_severity"),
                            ("factual_accuracy", "avg_factual_accuracy"),
                            ("overconfidence", "avg_overconfidence"),
                            ("overall_reliability", "avg_overall_reliability")
                        ]:
                            if metric in scores:
                                dataset_stats[key] += scores[metric]
                                hallucination_stats["overall"][key] += scores[metric]
                
                output_file.write("\n" + "-"*50 + "\n")
        else:
            # Process without GPT evaluation (for each example)
            for i in range(batch_size):
                global_idx = batch_idx * args.batch_size + i
                
                # Print results 
                if global_idx < 5 or global_idx % 20 == 0:
                    print("\n" + "-"*80)
                    print(f"Example {global_idx+1}")
                    print(f"Question: {questions[i]}")
                    print(f"Ground Truth: {ground_truths[i]}")
                    print(f"Generated Answer: {generated_answers[i]}")
                
                # Write to output file
                output_file.write(f"Example {global_idx+1}\n")
                output_file.write(f"Question: {questions[i]}\n")
                output_file.write(f"Ground Truth: {ground_truths[i]}\n")
                output_file.write(f"Generated Answer: {generated_answers[i]}\n")
                output_file.write("\n" + "-"*50 + "\n")
    
    # Print and save aggregate GPT metrics if available
    if args.evaluate_with_gpt:
        aggregate_metrics = gpt_evaluation.print_gpt_aggregate_metrics(gpt_evals)
        gpt_evaluation.write_gpt_aggregate_metrics(output_file, gpt_evals)
        
        # Calculate dataset averages
        if hallucination_stats is not None:
            dataset_stats = hallucination_stats["dataset_stats"][dataset_name]
            total = dataset_stats["total_evaluated"]
            if total > 0:
                for key in ["avg_hallucination_severity", "avg_factual_accuracy", 
                           "avg_overconfidence", "avg_overall_reliability"]:
                    dataset_stats[key] /= total
    
    # Close output file
    output_file.close()
    
    print(f"\nEvaluation complete for {dataset_name}. Results saved to {output_path}")
    return dataset_info


def evaluate_dataset_wrapper(args_dict):
    """Wrapper function for parallel dataset evaluation"""
    dataset_name = args_dict["dataset_name"]
    model = args_dict["model"]
    tokenizer = args_dict["tokenizer"]
    device = args_dict["device"]
    args = args_dict["args"]
    hallucination_stats = args_dict["hallucination_stats"]
    model_results_dir = args_dict["model_results_dir"]
    
    return dataset_name, evaluate_dataset(
        dataset_name, 
        model, 
        tokenizer, 
        device,
        args,
        hallucination_stats,
        model_results_dir
    )


def main():
    """Main evaluation function with LoRA weights"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine number of processes if auto-detect is enabled
    if args.num_processes == 0:
        args.num_processes = min(len(args.dataset), mp.cpu_count() - 1)
        args.num_processes = max(1, args.num_processes)  # Ensure at least 1 process
    
    # Determine model type and set up appropriate directory names
    if args.lora_weights:
        # Extract training type from the LoRA weights path
        training_type = ""
        if "consistency" in args.lora_weights.lower():
            training_type = "consistency"
        elif "finetuning" in args.lora_weights.lower():
            training_type = "finetuning"
        
        if training_type:
            model_type = f"-LoRA-{training_type}"
        else:
            model_type = "-LoRA"
            
        print(f"Loading model: {args.model} with {training_type} LoRA weights from: {args.lora_weights}")
    else:
        model_type = "-Base"
        print(f"Loading model: {args.model} (base model without LoRA)")
    
    # Set up model results directory with appropriate suffix
    model_results_dir = model_utils.setup_results_directory(args.model + model_type)
    
    # Load model and tokenizer with LoRA weights (or base model if no LoRA weights)
    model, tokenizer = load_model_with_lora(args.model, args.lora_weights, args.device)
    
    # Set evaluation mode for the model
    model.eval()
    
    # Set random seed for reproducibility
    utils.seed_everything(args.seed)
    
    # Create a dictionary to store hallucination statistics across all datasets
    hallucination_stats = {
        "dataset_stats": {},
        "overall": {
            "total_evaluated": 0,
            "hallucination_counts": {
                "INTRINSIC": 0,
                "EXTRINSIC": 0,
                "FACTUAL_ERROR": 0,
                "NONE": 0,
                "UNKNOWN": 0
            },
            "avg_hallucination_severity": 0,
            "avg_factual_accuracy": 0,
            "avg_overconfidence": 0, 
            "avg_overall_reliability": 0
        }
    }
    
    # Run evaluation on each dataset - either sequentially or in parallel
    dataset_infos = {}
    
    if args.parallel_datasets and len(args.dataset) > 1:
        print(f"Running {len(args.dataset)} datasets in parallel with {args.num_processes} processes")
        
        # Create task partitioner
        task_partitioner = TaskPartitioner(seed=args.seed)
        
        # Add tasks for each dataset
        for dataset_name in args.dataset:
            task_args = {
                "dataset_name": dataset_name,
                "model": model,
                "tokenizer": tokenizer,
                "device": args.device,
                "args": args,
                "hallucination_stats": hallucination_stats,
                "model_results_dir": model_results_dir
            }
            task_partitioner.add_task_with_key(dataset_name, evaluate_dataset_wrapper, task_args)
        
        # Run tasks in parallel
        results = task_partitioner.run_multi_process(nprocesses=args.num_processes, cache_only=False)
        
        # Collect results
        for dataset_name, result in results.items():
            dataset_infos[dataset_name] = result[1]  # result[0] is dataset_name, result[1] is the info
    else:
        # Run sequentially
        for dataset_name in args.dataset:
            dataset_infos[dataset_name] = evaluate_dataset(
                dataset_name, 
                model, 
                tokenizer, 
                args.device,
                args,
                hallucination_stats,
                model_results_dir
            )
    
    # Save dataset analysis to a file for reference
    results_suffix = model_type.lower().strip("-")
    with open(f"{model_results_dir}/dataset_test_results_{args.model}_{results_suffix}.json", "w") as f:
        json.dump(dataset_infos, f, indent=2)
    
    # Save hallucination statistics to a separate file
    if args.evaluate_with_gpt:
        # Calculate overall averages
        total = hallucination_stats["overall"]["total_evaluated"]
        if total > 0:
            for metric in ["avg_hallucination_severity", "avg_factual_accuracy", 
                          "avg_overconfidence", "avg_overall_reliability"]:
                hallucination_stats["overall"][metric] /= total
        
        # Save hallucination statistics with training type in filename
        with open(f"{model_results_dir}/hallucination_stats_{args.model}_{results_suffix}.json", "w") as f:
            json.dump(hallucination_stats, f, indent=2)
        
        # Print overall hallucination summary
        print("\n" + "="*80)
        print(f"OVERALL HALLUCINATION ANALYSIS FOR {args.model}{model_type}")
        print("="*80)
        print(f"Total examples evaluated: {total}")
        print("\nHallucination type distribution:")
        for h_type, count in hallucination_stats["overall"]["hallucination_counts"].items():
            if total > 0:
                percentage = (count / total) * 100
                print(f"  {h_type}: {count} ({percentage:.1f}%)")
        
        print("\nAverage metrics across all datasets:")
        print(f"  Hallucination Severity: {hallucination_stats['overall']['avg_hallucination_severity']:.2f}/10")
        print(f"  Factual Accuracy: {hallucination_stats['overall']['avg_factual_accuracy']:.2f}/10")
        print(f"  Overconfidence: {hallucination_stats['overall']['avg_overconfidence']:.2f}/10")
        print(f"  Overall Reliability: {hallucination_stats['overall']['avg_overall_reliability']:.2f}/10")
        print("="*80)


if __name__ == "__main__":
    main() 