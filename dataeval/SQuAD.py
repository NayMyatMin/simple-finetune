import functools
import json
import os
import requests
from pathlib import Path

import datasets
import pandas as pd
from datasets import Dataset

# Import _settings or define fallback paths
try:
    import _settings
    DATA_FOLDER = _settings.DATA_FOLDER
except ImportError:
    # Fallback to local paths if _settings module is not available
    print("Warning: _settings module not found, using local paths instead")
    DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'datasets')
    os.makedirs(DATA_FOLDER, exist_ok=True)

# GitHub URL for the SQuAD dev-v2.0.json file
SQUAD_URL = "https://raw.githubusercontent.com/chrischute/squad/master/data/dev-v2.0.json"

def _save_dataset():
    # Create the path for saving the dataset
    save_path = os.path.join(DATA_FOLDER, 'squad_dataset')
    squad_file_path = os.path.join(DATA_FOLDER, 'dev-v2.0.json')
    
    # If the processed dataset already exists, return the path
    if os.path.exists(save_path):
        return save_path
        
    # Create the datasets directory if it doesn't exist
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(squad_file_path):
        print(f"SQuAD dataset not found. Downloading from GitHub...")
        try:
            response = requests.get(SQUAD_URL)
            response.raise_for_status()  # Check for download errors
            
            # Save the downloaded file
            with open(squad_file_path, 'wb') as f:
                f.write(response.content)
            print(f"SQuAD dataset downloaded successfully to {squad_file_path}")
        except Exception as e:
            raise Exception(f"Failed to download SQuAD dataset: {str(e)}")
    
    # Now process the dataset
    try:
        with open(squad_file_path, 'r') as infile:
            data = json.load(infile)['data']
        
        dataset = {}
        dataset['context'] = []
        dataset['question'] = []
        dataset['answers'] = []
        dataset['id'] = []
        
        # Process the SQuAD JSON structure
        for article in data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    # Skip questions that are marked as impossible (unanswerable)
                    if 'is_impossible' in qa and qa['is_impossible']:
                        continue
                    
                    question = qa['question']
                    id = qa['id']
                    answers = qa['answers']
                    
                    # Skip if no answers are provided
                    if not answers:
                        continue
                    
                    dataset['context'].append(context)
                    dataset['question'].append(question)
                    dataset['answers'].append(answers)
                    dataset['id'].append(id)
        
        # Convert to pandas DataFrame and then to HuggingFace Dataset
        dataset_df = pd.DataFrame.from_dict(dataset)
        dataset = Dataset.from_pandas(dataset_df)
        
        print(f"SQuAD dataset processed and filtered to include only answerable questions. Total: {len(dataset)} examples")
        dataset.save_to_disk(save_path)
        print(f"SQuAD dataset processed and saved to {save_path}")
    except Exception as e:
        raise Exception(f"Error processing SQuAD dataset: {str(e)}")
        
    return save_path

@functools.lru_cache(1)
def read_all_contexts():
    dataset = datasets.load_from_disk(_save_dataset())
    return {_['id']: _['context'] for _ in dataset}

def tokenize_squad(examples, tokenizer):
    """Tokenize SQuAD dataset examples for input to the model"""
    prompts = []
    for i in range(len(examples['context'])):
        prompt = f"Context: {examples['context'][i]}\nQuestion: {examples['question'][i]}\nAnswer:"
        prompts.append(prompt)
    
    # Tokenize inputs
    tokenized_examples = tokenizer(prompts, truncation=False, padding=False)
    examples['prompt'] = prompts
    examples['input_ids'] = tokenized_examples['input_ids']
    examples['attention_mask'] = tokenized_examples['attention_mask']
    return examples

def get_dataset(tokenizer, split='validation'):
    """Load the SQuAD dataset and prepare it for the model"""
    # Load from processed cache or create new
    dataset = datasets.load_from_disk(_save_dataset())
    
    # Process into model inputs
    dataset = dataset.map(
        lambda examples: tokenize_squad(examples, tokenizer),
        batched=True,
        load_from_cache_file=False,
    )
    
    # Set format for model input
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], 
                      output_all_columns=True)
    return dataset

def _generate_config(tokenizer):
    """
    Configure generation parameters for the tokenizer.
    Works with LlamaTokenizerFast, LlamaTokenizer, and MistralTokenizerFast.
    """
    # Get end-of-sequence tokens
    try:
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]
    except:
        # Fallback encoding method
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['.', '\n']]
    
    # Add the model's EOS token
    eos_token_id.append(tokenizer.eos_token_id)
    
    # Prevent model from generating further questions
    try:
        bad_words_ids = [tokenizer.encode(_)[1:] for _ in ['Question:', '\nQuestion']]
    except:
        # Fallback for different tokenizer formats
        bad_words_ids = [tokenizer(_)['input_ids'][1:] for _ in ['Question:', '\nQuestion']]
    
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)

if __name__ == '__main__':
    # Simple test code
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    dataset = get_dataset(tokenizer)
    print(f"Loaded dataset with {len(dataset)} examples")
    print(f"Example prompt: {dataset[0]['prompt']}")