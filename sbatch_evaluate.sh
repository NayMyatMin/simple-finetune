#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # Use 1 node
#SBATCH --cpus-per-task=10          # Increase to 10 CPUs for faster processing
#SBATCH --mem=48GB                  # Increase memory to 48GB
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --constraint=a100           # Target A100 GPUs specifically
#SBATCH --time=02-00:00:00          # Maximum run time of 2 days
##SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications for job start, end, and failure
#SBATCH --output=%u.%x              # Log file location

################################################################
## EDIT %u.%j.out AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # Partition assigned
#SBATCH --account=sunjunresearch   # Account assigned (use myinfo command to check)
#SBATCH --qos=research-1-qos         # QOS assigned (use myinfo command to check)
#SBATCH --job-name=evaluate    # Default job name (will be overridden)
#SBATCH --mail-user=myatmin.nay.2022@phdcs.smu.edu.sg  # Email notifications

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Set job name based on model and dataset if provided
if [ -n "$MODEL_NAME" ] && [ -n "$DATASET_NAME" ]; then
    export SLURM_JOB_NAME="${MODEL_NAME}_${DATASET_NAME}"
fi

# Get the parent directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Python/3.10.16-GCCcore-13.3.0 
module load CUDA/12.6.0

source ~/myenv/bin/activate

# Extract API key from .bashrc - more robust method for both HF and OpenAI keys
BASHRC_PATH=~/.bashrc
if [ -f "$BASHRC_PATH" ]; then
    # Try to get Hugging Face token with various patterns to handle different formats
    HF_TOKEN_LINE=$(grep 'HUGGING_FACE_HUB_TOKEN' "$BASHRC_PATH" | tail -n 1)
    if [[ "$HF_TOKEN_LINE" == *"=\""* ]]; then
        HF_TOKEN=$(echo "$HF_TOKEN_LINE" | sed -E 's/.*="([^"]+)".*/\1/')
    elif [[ "$HF_TOKEN_LINE" == *"='"* ]]; then
        HF_TOKEN=$(echo "$HF_TOKEN_LINE" | sed -E "s/.*='([^']+)'.*/\1/")
    else
        HF_TOKEN=$(echo "$HF_TOKEN_LINE" | sed -E 's/.*=([^ ]+).*/\1/')
    fi
    
    # Try to get OpenAI API key with various patterns
    OPENAI_API_KEY_LINE=$(grep 'OPENAI_API_KEY' "$BASHRC_PATH" | tail -n 1)
    if [[ "$OPENAI_API_KEY_LINE" == *"=\""* ]]; then
        OPENAI_API_KEY=$(echo "$OPENAI_API_KEY_LINE" | sed -E 's/.*="([^"]+)".*/\1/')
    elif [[ "$OPENAI_API_KEY_LINE" == *"='"* ]]; then
        OPENAI_API_KEY=$(echo "$OPENAI_API_KEY_LINE" | sed -E "s/.*='([^']+)'.*/\1/")
    else
        OPENAI_API_KEY=$(echo "$OPENAI_API_KEY_LINE" | sed -E 's/.*=([^ ]+).*/\1/')
    fi
    
    # Export the tokens if found
    if [ -n "$HF_TOKEN" ]; then
        export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
        echo "Hugging Face token loaded"
    fi
    
    if [ -n "$OPENAI_API_KEY" ]; then
        export OPENAI_API_KEY=$OPENAI_API_KEY
        echo "OpenAI API key loaded"
    fi
fi

python evaluate.py --dataset coqa triviaqa nq_open SQuAD --fraction_of_data_to_use 0.001 --evaluate_with_gpt