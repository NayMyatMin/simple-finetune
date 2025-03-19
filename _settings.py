import os

# Base directory is the current directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_FOLDER = os.path.join(_BASE_DIR, 'data', 'datasets')
MODEL_PATH = os.path.join(_BASE_DIR, 'data', 'weights')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'data', 'output')

# Create directories if they don't exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(GENERATION_FOLDER, exist_ok=True)

print(f"_settings.py loaded. DATA_FOLDER={DATA_FOLDER}")

