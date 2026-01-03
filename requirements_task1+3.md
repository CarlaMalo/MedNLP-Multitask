conda create -n task3-nlp python=3.10 -y
conda activate task3-nlp

pip install -U pip

pip install \
  numpy \
  pandas \
  scikit-learn \
  scipy \
  matplotlib \
  seaborn

pip install \
  spacy \
  datasets \
  nltk

python -m spacy download en_core_web_md

pip install \
  transformers \
  accelerate \
  evaluate \
  sentencepiece \
  sentence-transformers==2.3.1 \
  sympy==1.13.3

pip install peft bitsandbytes

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
