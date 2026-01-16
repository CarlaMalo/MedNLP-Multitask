import torch
import torchvision
import spacy
from spacy_llm.util import assemble
import json
import os
import argparse
import warnings
from transformers import logging as transformers_logging

# Suppress transformers warnings about attention mask and pad token
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad token.*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and initialize spaCy-LLM model")
    # Models available: mistral-7b, dolly-v2-3b, Llama-2-13b-hf
    parser.add_argument("--model", type=str, default="mistral-7b", help="Model name to use (default: mistral_7b)")
    args = parser.parse_args()
    model = args.model

    # Assemble spaCy-LLM config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, f"configs/{model}.cfg")
    nlp = assemble(config_path)

    # Run the LLM pipeline (Frozen llm component with NER head)
    nlp.initialize()
    output_path = os.path.join(script_dir, f"models/output_{model}_ner")
    nlp.to_disk(output_path)
    print(f"Model saved to {output_path}")