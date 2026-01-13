#!/bin/bash
set -e

# Create and activate conda environment
conda create -n task3-nlp python=3.10 -y
# Use 'conda run' to ensure commands run in the environment
echo "Activating conda environment..."
CONDA_ENV_NAME="task3-nlp"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_NAME

# Upgrade pip
pip install --upgrade pip

# Install basic Python packages
pip install numpy pandas scikit-learn scipy matplotlib seaborn

# Install NLP-related packages
pip install spacy datasets nltk

# Download spaCy English model
python -m spacy download en_core_web_md

# Install transformers and related packages
pip install transformers accelerate evaluate sentencepiece sentence-transformers==2.3.1 sympy==1.13.3

# Install additional packages
pip install peft bitsandbytes

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Setup complete! Environment '$CONDA_ENV_NAME' is ready."
