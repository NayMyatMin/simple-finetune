#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import json
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

# Add parent directory to path to find _settings
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Handle _settings import
try:
    import _settings
except ImportError:
    # Create a simple _settings module if it doesn't exist
    class Settings:
        def __init__(self):
            self.DATA_FOLDER = os.path.join(parent_dir, 'data', 'datasets')
            # Create directories
            os.makedirs(self.DATA_FOLDER, exist_ok=True)
    
    # Create the settings object and add it to sys.modules
    _settings = Settings()
    sys.modules['_settings'] = _settings
    print("Created temporary _settings module")

# Import EigenScore modules - handle both module and direct imports
try:
    # Try relative imports (when imported as a module)
    from . import coqa
    from . import triviaqa
    from . import nq_open
    from . import SQuAD
    try:
        from . import TruthfulQA
        has_truthfulqa = True
    except ImportError:
        has_truthfulqa = False
except ImportError:
    # Fall back to direct imports (when run as a script)
    import dataeval.coqa as coqa
    import dataeval.triviaqa as triviaqa
    import dataeval.nq_open as nq_open
    import dataeval.SQuAD as SQuAD
    try:
        import dataeval.TruthfulQA as TruthfulQA
        has_truthfulqa = True
    except ImportError:
        has_truthfulqa = False
    except Exception as e:
        print(f"Error importing TruthfulQA: {e}")
        has_truthfulqa = False

class Llama2DatasetTester:
    def __init__(self, args):
        self.args = args
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        # Update log file path to be created in parent directory
        self.log_file = open(os.path.join(parent_dir, f"dataset_test_results_llama2_7b_chat.log"), "w")
        self.results = defaultdict(dict)
        
        # Load tokenizer
        print(f"Loading tokenizer for {self.model_name}...")
        try:
            # Try direct loading with AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Tokenizer loaded successfully using AutoTokenizer")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            sys.exit(1)
        
        # Define datasets to test
        self.datasets = {
            "coqa": coqa.get_dataset,
            "triviaqa": triviaqa.get_dataset,
            "nq_open": nq_open.get_dataset,
            "squad": SQuAD.get_dataset
        }
        
        # Add TruthfulQA if available
        if has_truthfulqa:
            self.datasets["truthfulqa"] = TruthfulQA.get_dataset
        
        # Filter datasets if specified
        if args.datasets:
            self.datasets = {k: v for k, v in self.datasets.items() if k in args.datasets}
    
    def log(self, message):
        """Write to both console and log file"""
        print(message)
        self.log_file.write(f"{message}\n")
        self.log_file.flush()
    
    def test_loading(self, dataset_name, dataset_fn):
        """Test dataset loading and basic properties"""
        self.log(f"\n{'='*50}")
        self.log(f"Testing {dataset_name.upper()} dataset loading with Llama-2-7b-chat-hf...")
        
        try:
            # Get dataset with fraction if specified
            if self.args.fraction:
                # We need to implement this separately since dataset loading doesn't have fraction parameter
                dataset = dataset_fn(self.tokenizer)
                dataset = dataset.train_test_split(test_size=self.args.fraction, seed=42)['train']
            else:
                dataset = dataset_fn(self.tokenizer)
            
            # Basic dataset properties
            self.log(f"Dataset loaded successfully with {len(dataset)} examples")
            self.results[dataset_name]["size"] = len(dataset)
            
            # Sample some examples
            samples = min(5, len(dataset))
            self.log(f"Sampling {samples} examples...")
            
            # Get column names
            if hasattr(dataset, 'column_names'):
                columns = dataset.column_names
            else:
                # Extract column names from first example
                columns = list(dataset[0].keys())
            
            self.log(f"Dataset columns: {columns}")
            self.results[dataset_name]["columns"] = columns
            
            # Analyze columns types and content
            column_analysis = {}
            for col in columns:
                try:
                    if col in ['input_ids', 'attention_mask']:
                        # Skip tokenizer outputs
                        continue
                    
                    sample_value = dataset[0][col]
                    value_type = type(sample_value).__name__
                    
                    if isinstance(sample_value, str):
                        avg_length = sum(len(dataset[i][col]) for i in range(min(100, len(dataset)))) / min(100, len(dataset))
                        column_analysis[col] = {
                            "type": value_type,
                            "avg_length": f"{avg_length:.2f} chars"
                        }
                    elif isinstance(sample_value, list) or isinstance(sample_value, torch.Tensor):
                        avg_length = sum(len(dataset[i][col]) for i in range(min(100, len(dataset)))) / min(100, len(dataset))
                        column_analysis[col] = {
                            "type": value_type,
                            "avg_length": f"{avg_length:.2f} elements"
                        }
                    else:
                        column_analysis[col] = {"type": value_type}
                except Exception as e:
                    column_analysis[col] = {"type": "error", "error": str(e)}
            
            self.log(f"Column analysis: {json.dumps(column_analysis, indent=2)}")
            self.results[dataset_name]["column_analysis"] = column_analysis
            
            # Check for prompt format
            if 'prompt' in columns:
                self.log("\nPrompt format examples:")
                for i in range(min(3, len(dataset))):
                    self.log(f"Example {i+1}:\n{dataset[i]['prompt']}\n")
                
                # Analyze prompt lengths
                prompt_lengths = [len(dataset[i]['prompt']) for i in range(min(1000, len(dataset)))]
                avg_prompt_len = sum(prompt_lengths) / len(prompt_lengths)
                max_prompt_len = max(prompt_lengths)
                min_prompt_len = min(prompt_lengths)
                
                self.log(f"Prompt length stats: Avg={avg_prompt_len:.2f}, Min={min_prompt_len}, Max={max_prompt_len}")
                self.results[dataset_name]["prompt_stats"] = {
                    "avg_length": avg_prompt_len,
                    "min_length": min_prompt_len,
                    "max_length": max_prompt_len
                }
            
            # Check for tokenization results
            if 'input_ids' in columns:
                tokenization_stats = self.analyze_tokenization(dataset, dataset_name)
                self.results[dataset_name]["tokenization"] = tokenization_stats
            
            # Test generation config
            try:
                # Get the current module
                module = None
                if dataset_name.lower() == "coqa":
                    module = coqa
                elif dataset_name.lower() == "triviaqa":
                    module = triviaqa
                elif dataset_name.lower() == "nq_open":
                    module = nq_open
                elif dataset_name.lower() == "squad":
                    module = SQuAD
                elif dataset_name.lower() == "truthfulqa" and has_truthfulqa:
                    module = TruthfulQA
                
                if module and hasattr(module, "_generate_config"):
                    generation_config = getattr(module, "_generate_config")(self.tokenizer)
                    self.log(f"\nGeneration config: {json.dumps(generation_config, default=str, indent=2)}")
                    self.results[dataset_name]["generation_config"] = generation_config
            except Exception as e:
                self.log(f"Error getting generation config: {str(e)}")
            
            return True
            
        except Exception as e:
            self.log(f"Error loading {dataset_name} dataset: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.results[dataset_name]["error"] = str(e)
            return False
    
    def analyze_tokenization(self, dataset, dataset_name):
        """Analyze tokenization results for the dataset"""
        self.log("\nAnalyzing tokenization...")
        
        # Get input_ids lengths
        input_lengths = [len(dataset[i]['input_ids']) for i in range(min(1000, len(dataset)))]
        avg_length = sum(input_lengths) / len(input_lengths)
        max_length = max(input_lengths)
        min_length = min(input_lengths)
        
        self.log(f"Token length stats: Avg={avg_length:.2f}, Min={min_length}, Max={max_length}")
        
        # Check for potential truncation issues
        potential_truncation = sum(1 for l in input_lengths if l > 2048)
        self.log(f"Examples potentially needing truncation (>2048 tokens): {potential_truncation} ({potential_truncation/len(input_lengths):.2%})")
        
        # Decode a sample to verify correct tokenization
        sample_idx = min(10, len(dataset)-1)
        original_ids = dataset[sample_idx]['input_ids']
        decoded_text = self.tokenizer.decode(original_ids)
        
        self.log(f"\nSample tokenization verification (example {sample_idx}):")
        self.log(f"Original input: {dataset[sample_idx].get('prompt', 'N/A')}")
        self.log(f"Decoded from tokens: {decoded_text}")
        
        # Create histogram of input lengths if plotting enabled
        if self.args.plot:
            plot_path = os.path.join(parent_dir, f"llama2_{dataset_name}_token_distribution.png")
            plt.figure(figsize=(10, 6))
            plt.hist(input_lengths, bins=50)
            plt.title(f"{dataset_name.upper()} - Input Token Length Distribution (Llama 2)")
            plt.xlabel("Token Count")
            plt.ylabel("Frequency")
            plt.savefig(plot_path)
            plt.close()
            self.log(f"Saved token distribution plot to {plot_path}")
        
        return {
            "avg_length": avg_length,
            "min_length": min_length,
            "max_length": max_length,
            "potential_truncation_percent": potential_truncation/len(input_lengths)
        }
    
    def run_all_tests(self):
        """Run tests for all datasets with Llama 2 tokenizer"""
        self.log(f"Starting dataset tests with model: {self.model_name}")
        self.log(f"Testing datasets: {', '.join(self.datasets.keys())}")
        
        successful = 0
        for name, dataset_fn in self.datasets.items():
            if self.test_loading(name, dataset_fn):
                successful += 1
        
        self.log(f"\n{'='*50}")
        self.log(f"Testing complete: {successful}/{len(self.datasets)} datasets loaded successfully")
        
        # Generate summary report
        self.generate_report()
        
        self.log_file.close()
    
    def generate_report(self):
        """Generate a summary report of test results"""
        self.log("\nGENERATING SUMMARY REPORT FOR LLAMA 2 TESTING")
        self.log("-" * 50)
        
        # Dataset size comparison
        sizes = {name: data.get("size", 0) for name, data in self.results.items()}
        self.log(f"Dataset sizes: {json.dumps(sizes, indent=2)}")
        
        # Token length comparison
        token_stats = {name: data.get("tokenization", {}) for name, data in self.results.items() if "tokenization" in data}
        token_summary = {}
        for name, stats in token_stats.items():
            token_summary[name] = {
                "avg_tokens": stats.get("avg_length", "N/A"),
                "max_tokens": stats.get("max_length", "N/A")
            }
        
        self.log(f"Tokenization summary: {json.dumps(token_summary, indent=2)}")
        
        # Save results to JSON for further analysis
        json_path = os.path.join(parent_dir, f"dataset_test_results_llama2_7b_chat_hf.json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.log(f"Detailed results saved to {json_path}")
        
        # Create comparison table if multiple datasets were tested
        if len(self.results) > 1:
            comparison_data = []
            for name, data in self.results.items():
                row = {
                    "Dataset": name,
                    "Size": data.get("size", "N/A"),
                    "Avg Tokens": data.get("tokenization", {}).get("avg_length", "N/A"),
                    "Max Tokens": data.get("tokenization", {}).get("max_length", "N/A"),
                    "Prompt Columns": "Yes" if "prompt" in data.get("columns", []) else "No",
                    "Gen Config": "Yes" if "generation_config" in data else "No"
                }
                comparison_data.append(row)
            
            # Convert to dataframe for nice display
            df = pd.DataFrame(comparison_data)
            self.log("\nDataset Comparison for Llama 2:")
            self.log(df.to_string())

def main():
    parser = argparse.ArgumentParser(description="Test datasets with Llama-2-7b-chat tokenizer")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to test (default: all)")
    parser.add_argument("--fraction", type=float, help="Fraction of dataset to use (0-1)")
    parser.add_argument("--plot", action="store_true", help="Generate plots for analysis")
    parser.add_argument("--squad", action="store_true", help="Test only SQuAD dataset")
    
    args = parser.parse_args()
    
    if args.squad:
        # Only test SQuAD dataset
        args.datasets = ["squad"]
    
    tester = Llama2DatasetTester(args)
    tester.run_all_tests()

if __name__ == "__main__":
    main() 