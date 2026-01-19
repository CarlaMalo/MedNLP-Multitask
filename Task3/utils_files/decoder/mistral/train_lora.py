from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from datasets import Dataset


def build_sft_dataset(texts, labels, prompt_fn, tokenizer, label_list):
    """Construct dataset with prompt + response concatenated.

    We build a single sequence: [prompt tokens][label + eos]. Loss is only
    applied on the response portion. Prompt tokens are masked with -100.
    """

    prompts, responses = [], []

    for t, l in zip(texts, labels):
        prompt = prompt_fn(t, label_list)
        response = l + tokenizer.eos_token
        prompts.append(prompt)
        responses.append(response)

    return Dataset.from_dict({
        "prompt": prompts,
        "response": responses
    })


def tokenize_sft(batch, tokenizer, max_length=512):
    """Tokenize SFT dataset with proper masking for prompt/response.
    
    Ensures response is never truncated by reserving space for it.
    """

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = []
    labels = []
    attention_masks = []
    
    min_response_length = 10  # Reserve minimum space for response

    for prompt_text, response_text in zip(batch["prompt"], batch["response"]):
        # Tokenize prompt and response separately with consistent settings
        prompt_tokens = tokenizer(
            prompt_text,
            padding=False,
            truncation=False,
            add_special_tokens=True
        )
        
        response_tokens = tokenizer(
            response_text,
            padding=False,
            truncation=False,
            add_special_tokens=False  # Response doesn't need BOS, already in prompt
        )
        
        prompt_ids = prompt_tokens["input_ids"]
        response_ids = response_tokens["input_ids"]
        
        # Ensure response fits; truncate prompt if needed
        if len(prompt_ids) + len(response_ids) > max_length:
            max_prompt_len = max(max_length - len(response_ids), min_response_length)
            prompt_ids = prompt_ids[:max_prompt_len]
        
        # Concatenate
        combined = prompt_ids + response_ids
        
        # Truncate if still too long 
        if len(combined) > max_length:
            combined = combined[:max_length]
        
        # Create labels: mask prompt tokens, keep response tokens
        prompt_len = len(prompt_ids)
        label_seq = [-100] * prompt_len + combined[prompt_len:]
        
        # Pad to max_length
        pad_len = max_length - len(combined)
        if pad_len > 0:
            combined = combined + [pad_id] * pad_len
            label_seq = label_seq + ([-100] * pad_len)
            attn = [1] * (max_length - pad_len) + [0] * pad_len
        else:
            attn = [1] * max_length

        input_ids.append(combined)
        labels.append(label_seq)
        attention_masks.append(attn)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_masks}

def train_lora(
    model,
    tokenizer,
    train_texts,
    train_labels,
    label_list,
    prompt_fn,
    output_dir="Task3/mistral_lora",
    eval_split=0.1
):
    """Train LoRA SFT on the provided training data with validation split.
    
    Args:
        eval_split: Proportion of training data to use for validation (default 0.1)
    """
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    train_ds = build_sft_dataset(
        train_texts, train_labels, prompt_fn, tokenizer, label_list
    )
    train_ds = train_ds.map(
        lambda b: tokenize_sft(b, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
        load_from_cache_file=True,
        desc="Tokenizing dataset"
    )
    
    # Split into train and eval
    split_ds = train_ds.train_test_split(test_size=eval_split, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=8,
        save_total_limit=1,
        fp16=True,
        save_strategy="epoch", # "no"
        eval_strategy="epoch",
        logging_strategy="epoch",  
        report_to="none",
        disable_tqdm=True,
        log_level="error",  # Set to 'error' to suppress warnings
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds
    )

    trainer.train()
    return model
