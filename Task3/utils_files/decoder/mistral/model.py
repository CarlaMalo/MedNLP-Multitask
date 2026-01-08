import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

def load_mistral(load_in_4bit=True):
    """ Load Mistral model and tokenizer with optional 4-bit quantization. """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16
    ) if load_in_4bit else None
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16,
        quantization_config=quantization_config
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        # For warnings about no pad_token_id in generation
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()

    return model, tokenizer