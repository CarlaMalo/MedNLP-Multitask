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
    """  Tokenize SFT dataset with proper masking for prompt/response."""

    # Tokenize prompt and response separately to know boundaries
    prompt_tokens = tokenizer(
        batch["prompt"],
        padding=False,
        truncation=True,
        max_length=max_length
    )

    response_tokens = tokenizer(
        batch["response"],
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = []
    labels = []
    attention_masks = []

    for p_ids, r_ids in zip(prompt_tokens["input_ids"], response_tokens["input_ids"]):
        # Truncate if combined is too long
        combined = (p_ids + r_ids)[:max_length]

        # Labels: mask prompt tokens with -100 so loss only on response part
        label_seq = ([-100] * len(p_ids) + r_ids)[:max_length]

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
    output_dir="Task3/mistral_lora"
):
    """Train LoRA SFT on the provided training data."""
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

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=15,
        fp16=True,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True,
        disable_tqdm=True,
        log_level="warning"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds
    )

    trainer.train()
    return model
