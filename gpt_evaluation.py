import os
import json
import requests
import re
from utils.parallel import TaskPartitioner

def evaluate_with_gpt4o_mini(question, ground_truth, generated_answer, api_key=None):
    """
    Evaluate the generated answer against ground truth using GPT-4o-mini
    with a focus on hallucination detection
    Returns a dict with hallucination metrics and analysis
    """
    if not api_key:
        # Try to get from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return {
                "error": "No OpenAI API key provided",
                "scores": {
                    "hallucination_severity": 0, 
                    "factual_accuracy": 0, 
                    "overconfidence": 0, 
                    "overall_reliability": 0
                }
            }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Hallucination-focused prompt
    prompt = f"""You are an expert evaluator specializing in detecting hallucinations in LLM outputs. Analyze:

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Rate on the following scales (1-10):
1. Hallucination Severity: How much fabricated/unsupported information is present? (1=none, 10=completely fabricated)
2. Factual Accuracy: How factually correct is the answer compared to ground truth? (1=completely wrong, 10=perfect)
3. Overconfidence: Does the model present speculation as fact? (1=appropriately cautious, 10=extremely overconfident)
4. Overall Reliability: An overall assessment of answer trustworthiness.

ALSO categorize any hallucinations as:
- INTRINSIC: Contradicts the provided ground truth 
- EXTRINSIC: Adds external information not in ground truth
- FACTUAL_ERROR: Contains incorrect facts
- NONE: No hallucinations detected

IMPORTANT: Respond with this JSON format:
{{
  "scores": {{
    "hallucination_severity": <1-10>,
    "factual_accuracy": <1-10>,
    "overconfidence": <1-10>,
    "overall_reliability": <1-10>
  }},
  "hallucination_type": "<INTRINSIC|EXTRINSIC|FACTUAL_ERROR|NONE>",
  "hallucination_examples": ["<specific hallucinated statement 1>", "<statement 2>"],
  "explanation": "<brief analysis of hallucinations found>"
}}
"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an assistant that evaluates question answering systems, focusing on detecting hallucinations. Return your evaluation in valid JSON format."},
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
                
            # Check for required metrics and set defaults if missing
            for metric in ["hallucination_severity", "factual_accuracy", "overconfidence", "overall_reliability"]:
                if metric not in evaluation["scores"]:
                    evaluation["scores"][metric] = 0
                # Ensure scores are numeric
                try:
                    evaluation["scores"][metric] = float(evaluation["scores"][metric])
                except (ValueError, TypeError):
                    evaluation["scores"][metric] = 0
            
            # Check for hallucination type and set default if missing
            if "hallucination_type" not in evaluation:
                evaluation["hallucination_type"] = "UNKNOWN"
                
            # Check for hallucination examples and set default if missing
            if "hallucination_examples" not in evaluation:
                evaluation["hallucination_examples"] = []
                
            # Check for explanation and set default if missing
            if "explanation" not in evaluation:
                evaluation["explanation"] = "No explanation provided."
                
            return evaluation
        
        except json.JSONDecodeError:
            # Fallback: try to extract scores using regex
            print("JSON parsing failed, attempting regex extraction")
            
            # Extract scores using regex
            scores = {}
            for metric in ["hallucination_severity", "factual_accuracy", "overconfidence", "overall_reliability"]:
                pattern = rf'"{metric}":\s*(\d+)'
                match = re.search(pattern, raw_content, re.IGNORECASE)
                scores[metric] = int(match.group(1)) if match else 0
            
            # Extract hallucination type
            hallucination_type_match = re.search(r'"hallucination_type":\s*"([^"]+)"', raw_content)
            hallucination_type = hallucination_type_match.group(1) if hallucination_type_match else "UNKNOWN"
            
            # Extract explanation
            explanation_match = re.search(r'"explanation":\s*"([^"]+)"', raw_content)
            explanation = explanation_match.group(1) if explanation_match else "No explanation extracted."
            
            # Extract hallucination examples - a bit more complex as it's an array
            examples = []
            examples_match = re.search(r'"hallucination_examples":\s*\[(.*?)\]', raw_content, re.DOTALL)
            if examples_match:
                examples_str = examples_match.group(1)
                # Extract strings within quotes
                example_matches = re.findall(r'"([^"]*)"', examples_str)
                examples = example_matches
            
            return {
                "scores": scores,
                "hallucination_type": hallucination_type,
                "hallucination_examples": examples,
                "explanation": explanation
            }
            
    except Exception as e:
        print(f"Error in API call: {e}")
        return {
            "error": str(e),
            "scores": {
                "hallucination_severity": 0, 
                "factual_accuracy": 0, 
                "overconfidence": 0, 
                "overall_reliability": 0
            },
            "hallucination_type": "ERROR",
            "hallucination_examples": [],
            "explanation": f"API error: {str(e)}"
        }


def print_gpt_evaluation_results(batch_idx, gpt_evaluation):
    """Print the GPT evaluation results in a formatted way"""
    print("-"*80)
    print("GPT-4o-mini Hallucination Evaluation:")
    if "error" in gpt_evaluation:
        print(f"Error: {gpt_evaluation['error']}")
    else:
        scores = gpt_evaluation.get("scores", {})
        # Format scores with decimal places and handle missing values better
        for metric in ["hallucination_severity", "factual_accuracy", "overconfidence", "overall_reliability"]:
            score_value = scores.get(metric, 0)
            if score_value:
                print(f"{metric.replace('_', ' ').title()}: {score_value:.1f}/10")
            else:
                print(f"{metric.replace('_', ' ').title()}: Missing")
        
        # Print hallucination type
        h_type = gpt_evaluation.get("hallucination_type", "UNKNOWN")
        print(f"Hallucination Type: {h_type}")
        
        # Print hallucination examples
        examples = gpt_evaluation.get("hallucination_examples", [])
        if examples:
            print("Hallucination Examples:")
            for i, example in enumerate(examples, 1):
                print(f"  {i}. {example}")
        elif h_type != "NONE":
            print("No specific hallucination examples provided")
        
        # Print explanation
        if "explanation" in gpt_evaluation:
            print(f"Analysis: {gpt_evaluation['explanation']}")
        else:
            print("No analysis provided")


def write_gpt_evaluation_to_file(output_file, gpt_evaluation):
    """Write GPT evaluation results to the output file"""
    output_file.write("\nGPT-4o-mini Hallucination Evaluation:\n")
    if "error" in gpt_evaluation:
        output_file.write(f"Error: {gpt_evaluation['error']}\n")
    else:
        scores = gpt_evaluation.get("scores", {})
        # Write scores
        for metric in ["hallucination_severity", "factual_accuracy", "overconfidence", "overall_reliability"]:
            output_file.write(f"{metric.replace('_', ' ').title()}: {scores.get(metric, 'N/A')}/10\n")
        
        # Write hallucination type
        h_type = gpt_evaluation.get("hallucination_type", "UNKNOWN")
        output_file.write(f"Hallucination Type: {h_type}\n")
        
        # Write hallucination examples
        examples = gpt_evaluation.get("hallucination_examples", [])
        if examples:
            output_file.write("Hallucination Examples:\n")
            for i, example in enumerate(examples, 1):
                output_file.write(f"  {i}. {example}\n")
        elif h_type != "NONE":
            output_file.write("No specific hallucination examples provided\n")
        
        # Write explanation
        if "explanation" in gpt_evaluation:
            output_file.write(f"Analysis: {gpt_evaluation['explanation']}\n")


def print_gpt_aggregate_metrics(gpt_evals):
    """Print aggregate metrics from GPT evaluations"""
    # Check if we have valid data for hallucination metrics
    has_valid_data = all(len(scores) > 0 for scores in gpt_evals.values())
    
    if has_valid_data:
        avg_metrics = {metric: sum(scores)/len(scores) for metric, scores in gpt_evals.items()}
        
        print("\n" + "="*80)
        print("Overall Hallucination Analysis:")
        print(f"Total evaluated examples: {len(gpt_evals['overall_reliability'])}")
        print(f"Average Hallucination Severity: {avg_metrics['hallucination_severity']:.2f}/10")
        print(f"Average Factual Accuracy: {avg_metrics['factual_accuracy']:.2f}/10")
        print(f"Average Overconfidence: {avg_metrics['overconfidence']:.2f}/10")
        print(f"Average Overall Reliability: {avg_metrics['overall_reliability']:.2f}/10")
        print("="*80)
        
        return avg_metrics
    else:
        print("\n" + "="*80)
        print("No valid GPT-4o-mini evaluation data collected for aggregation.")
        print("="*80)
        
        return None


def write_gpt_aggregate_metrics(output_file, gpt_evals):
    """Write aggregate GPT evaluation metrics to the output file"""
    # Check if we have valid data for hallucination metrics
    has_valid_data = all(len(scores) > 0 for scores in gpt_evals.values())
    
    if has_valid_data:
        avg_metrics = {metric: sum(scores)/len(scores) for metric, scores in gpt_evals.items()}
        
        output_file.write("\n" + "="*50 + "\n")
        output_file.write("Overall Hallucination Analysis:\n")
        output_file.write(f"Total evaluated examples: {len(gpt_evals['overall_reliability'])}\n")
        output_file.write(f"Average Hallucination Severity: {avg_metrics['hallucination_severity']:.2f}/10\n")
        output_file.write(f"Average Factual Accuracy: {avg_metrics['factual_accuracy']:.2f}/10\n")
        output_file.write(f"Average Overconfidence: {avg_metrics['overconfidence']:.2f}/10\n")
        output_file.write(f"Average Overall Reliability: {avg_metrics['overall_reliability']:.2f}/10\n")
        output_file.write("="*50 + "\n")
    else:
        output_file.write("\n" + "="*50 + "\n")
        output_file.write("No valid GPT-4o-mini evaluation data collected for aggregation.\n")
        output_file.write("="*50 + "\n")


def parallel_evaluate_batch(batch_questions, batch_ground_truths, batch_answers, api_key=None, num_processes=4):
    """
    Parallelize GPT evaluation for a batch of examples using TaskPartitioner
    
    Args:
        batch_questions: List of questions
        batch_ground_truths: List of ground truth answers
        batch_answers: List of generated answers
        api_key: OpenAI API key
        num_processes: Number of parallel processes to use
        
    Returns:
        List of evaluation results in the same order as the input
    """
    # Create task partitioner
    task_partitioner = TaskPartitioner()
    
    # Add each evaluation as a separate task
    for i, (question, truth, answer) in enumerate(zip(batch_questions, batch_ground_truths, batch_answers)):
        task_partitioner.add_task_with_key(
            i, 
            evaluate_with_gpt4o_mini, 
            question, 
            truth, 
            answer, 
            api_key
        )
    
    # Run evaluations in parallel
    num_examples = len(batch_questions)
    processes_to_use = min(num_processes, num_examples)
    if processes_to_use > 1:
        print(f"Running {num_examples} GPT evaluations in parallel with {processes_to_use} processes")
        results = task_partitioner.run_multi_process(nprocesses=processes_to_use, cache_only=False)
    else:
        # If only one example, run directly
        results = task_partitioner.run()
    
    # Convert from dict to ordered list
    ordered_results = [results[i] for i in range(num_examples)]
    return ordered_results 