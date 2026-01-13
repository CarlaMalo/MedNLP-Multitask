#!/bin/bash
set -e

# Create and activate conda environment
CONDA_ENV_NAME="task2-nlp"
conda create -n $CONDA_ENV_NAME python=3.10 -y

echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_NAME

# Install core packages via conda
conda install -c conda-forge spacy -y
conda install numpy pandas scikit-learn scipy matplotlib seaborn -y
conda install datasets nltk tqdm ipykernel -y
conda install transformers accelerate evaluate sentencepiece sentence-transformers sympy -y
conda install peft pytorch torchvision torchaudio -y

# Install remaining packages via pip
pip install spacy-llm bitsandbytes

echo "Setup complete! Environment '$CONDA_ENV_NAME' is ready."
