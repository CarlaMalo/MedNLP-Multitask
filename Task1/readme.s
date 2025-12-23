Version pinning is required because newer versions of transformers/huggingface-hub
are not compatible with sentence-transformers on CPU environments.

1- This script creates a clean CPU-compatible environment to run Task 1 experiments.
conda create -n task1_nlp python=3.11 -y
conda activate task1_nlp
pip install pandas numpy matplotlib tqdm spacy==3.8.2
pip install scikit-learn==1.4.1.post1
pip install sentence-transformers==2.3.1
pip install transformers==4.35.2
pip install huggingface-hub==0.20.3
pip install tokenizers==0.15.2
pip install datasets==2.15.0
pip install seaborn
python -m spacy download en_core_web_sm
python -m ipykernel install --user --name task1_nlp --display-name "task1_nlp"

2- Other users can reproduce the environment using:
conda env create -f environment.yml