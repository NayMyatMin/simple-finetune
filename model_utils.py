import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import models

# Mapping from simplified model names to full HuggingFace paths
MODEL_PATH_MAPPING = {
    'Llama-2-7b-chat-hf': 'meta-llama/Llama-2-7b-chat-hf',
    'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3'
}


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


def load_model_from_hub(model_name, device):
    """Load model and tokenizer directly from HuggingFace Hub"""
    try:
        print(f"Attempting to load {model_name} directly from HuggingFace Hub...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        print(f"Successfully loaded {model_name} from HuggingFace Hub")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading from HuggingFace Hub: {e}")
        return None, None


def load_model(model_name, device, use_hf_directly=True):
    """Load model and tokenizer with fallback options"""
    short_model_name = model_name
    full_model_name = MODEL_PATH_MAPPING.get(model_name, model_name)
    
    if use_hf_directly:
        # Try loading directly from HuggingFace first
        model, tokenizer = load_model_from_hub(full_model_name, device)
        if model is not None:
            return model, tokenizer
        
        print(f"Falling back to local model loading with name: {short_model_name}")
    
    # Use the regular loading method from models module with the local name
    print(f"Loading model from local path using name: {short_model_name}")
    return models.load_model_and_tokenizer(short_model_name, device)


def extract_ground_truth(batch, dataset_name):
    """Extract ground truth answer from batch based on dataset format"""
    if dataset_name == 'SQuAD' and 'answers' in batch:
        # SQuAD dataset has answers in a different format
        try:
            # Check if the answers field is empty first
            if len(batch['answers']) > 0:
                answers = batch['answers'][0]  # Get the first element (list of answer dictionaries)
                
                # Print debug information (only for first few examples)
                if 'id' in batch and batch['id'][0].startswith('5726'):  # Just sample a few for debugging
                    print(f"SQuAD answers format: {type(answers)}")
                    print(f"SQuAD answers content: {answers}")
                
                # Handle the SQuAD format where answers is a list of dictionaries
                if isinstance(answers, list) and len(answers) > 0:
                    # Each answer is a dict with 'text' and 'answer_start'
                    if isinstance(answers[0], dict) and 'text' in answers[0]:
                        return answers[0]['text']  # Return the text of the first answer
                
                # Old logic for backwards compatibility
                elif isinstance(answers, dict) and 'text' in answers:
                    if isinstance(answers['text'], (list, tuple)) and len(answers['text']) > 0:
                        return answers['text'][0]  # Take the first answer text
                    elif isinstance(answers['text'], str):
                        return answers['text']  # Direct string
                
                # If no valid format was found
                if 'id' in batch and batch['id'][0].startswith('5726'):  # Just sample a few for debugging
                    print("Warning: SQuAD answers format is unexpected, couldn't extract ground truth")
            else:
                print("Warning: SQuAD example has empty answers field")
        except Exception as e:
            print(f"Error extracting SQuAD answer: {e}")
    elif 'answer' in batch:
        return batch['answer'][0]
    
    return None


def setup_results_directory(model_name):
    """Set up the results directory structure for the model"""
    results_dir = "sft_results"
    os.makedirs(results_dir, exist_ok=True)

    model_folder_name = model_name
    model_results_dir = os.path.join(results_dir, model_folder_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    return model_results_dir


def get_output_path(model_results_dir, dataset_name, custom_output_file=None):
    """Get the output file path for the results"""
    if custom_output_file:
        # User provided custom output path - append dataset name
        return f"{custom_output_file}_{dataset_name}"
    else:
        # Use standard format: results/MODEL_NAME/result_DATASET.txt
        return os.path.join(model_results_dir, f"result_{dataset_name}.txt") 