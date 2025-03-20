import argparse
import os
import torch
import tqdm
import transformers
import json
from datasets import Dataset

import _settings
import model_utils
import gpt_evaluation
import utils


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

    return parser.parse_args()


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
def evaluate_dataset(dataset_name, model, tokenizer, device, args, hallucination_stats=None):
    """Evaluate a single dataset using the provided model and tokenizer"""
    print(f"\n{'='*50}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*50}\n")
    
    # Set up model results directory
    model_results_dir = model_utils.setup_results_directory(args.model)
    
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
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
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
        
        # Generate response
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        
        # Decode the generated text after the prompt
        full_output_tokens = output[0]
        prompt_length = len(input_ids[0])
        generated_tokens = full_output_tokens[prompt_length:]
        
        generated_answer = tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        ).strip()
        
        # Extract original question and ground truth from batch
        raw_question = batch['question'][0]
        ground_truth = model_utils.extract_ground_truth(batch, dataset_name)
        
        # Print results 
        if batch_idx < 5 or batch_idx % 20 == 0:  # Print first 5 and then every 20th
            print("\n" + "-"*80)
            print(f"Example {batch_idx+1}")
            print(f"Question: {raw_question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Generated Answer: {generated_answer}")
        
        # Write to output file
        output_file.write(f"Example {batch_idx+1}\n")
        output_file.write(f"Question: {raw_question}\n")
        output_file.write(f"Ground Truth: {ground_truth}\n")
        output_file.write(f"Generated Answer: {generated_answer}\n")
        
        # Evaluate with GPT if enabled
        if args.evaluate_with_gpt:
            gpt_evaluation_result = gpt_evaluation.evaluate_with_gpt4o_mini(
                raw_question, 
                ground_truth, 
                generated_answer,
                api_key=args.openai_api_key
            )
            
            # Display and write GPT evaluation
            if batch_idx < 5 or batch_idx % 20 == 0:
                gpt_evaluation.print_gpt_evaluation_results(batch_idx, gpt_evaluation_result)
            
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


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load model and tokenizer once for all datasets
    print(f"Loading model: {args.model} on device: {args.device}")
    model, tokenizer = model_utils.load_model(args.model, args.device)
    
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
    
    # Run evaluation on each dataset
    dataset_infos = {}
    for dataset_name in args.dataset:
        dataset_infos[dataset_name] = evaluate_dataset(
            dataset_name, 
            model, 
            tokenizer, 
            args.device,
            args,
            hallucination_stats
        )
    
    # Save dataset analysis to a file for reference
    with open(f"results/{args.model}/dataset_test_results_{args.model}.json", "w") as f:
        json.dump(dataset_infos, f, indent=2)
    
    # Save hallucination statistics to a separate file
    if args.evaluate_with_gpt:
        # Calculate overall averages
        total = hallucination_stats["overall"]["total_evaluated"]
        if total > 0:
            for metric in ["avg_hallucination_severity", "avg_factual_accuracy", 
                          "avg_overconfidence", "avg_overall_reliability"]:
                hallucination_stats["overall"][metric] /= total
        
        # Save hallucination statistics
        with open(f"results/{args.model}/hallucination_stats_{args.model}.json", "w") as f:
            json.dump(hallucination_stats, f, indent=2)
        
        # Print overall hallucination summary
        print("\n" + "="*80)
        print(f"OVERALL HALLUCINATION ANALYSIS FOR {args.model}")
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
