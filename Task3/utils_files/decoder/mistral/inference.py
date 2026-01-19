import torch
from tqdm import tqdm
import re
from transformers import PrefixConstrainedLogitsProcessor

swap_pattern = re.compile(r'^\s*([^/]+?)\s*/\s*([^/]+?)\s*$')

# normalize_label not used currently since prefix constraint ensures valid labels
def normalize_label(pred, label_set):
    if pred is None:
        return "INVALID"

    pred = pred.strip()
    pred = pred.split("\n")[0]

    # Remove leading numbering
    pred = re.sub(r"^\d+\.\s*", "", pred)

    # Remove trailing punctuation
    pred = pred.rstrip(".,;:")

    # Remove explanatory text after label
    for label in label_set:
        if pred.startswith(label):
            return label

    # Handle "or" cases: pick first valid label
    if " or " in pred.lower():
        for label in label_set:
            if label.lower() in pred.lower():
                return label

    # Case-insensitive exact match
    for label in label_set:
        if pred.lower() == label.lower():
            return label
        
    # Handle swapped labels ("Obstetrics / Gynecology" <-> "Gynecology / Obstetrics")
    m = swap_pattern.match(pred)
    if m:
        swapped = f"{m.group(2).strip()} / {m.group(1).strip()}"
        if swapped in label_set:
            return swapped
    
    # Check if any label is contained in the prediction
    for label in label_set:
        if label.lower() in pred.lower():
            return label
    
    return "INVALID"


def build_label_token_map(labels, tokenizer):
    """
    Returns a list of token-id sequences, one per label.
    """
    label_token_ids = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        label_token_ids.append(ids)
    return label_token_ids

def make_allowed_tokens_fn(label_token_ids, eos_token_id, prompt_len):
    """
    Creates a prefix constraint function for HF generation.
    """

    def allowed_tokens_fn(batch_id, input_ids):
        # input_ids: tensor of shape [seq_len]
        generated = input_ids[prompt_len:].tolist()

        allowed = set()

        for label_ids in label_token_ids:
            
            if generated == label_ids[:len(generated)]:
                # If label is complete
                if len(generated) == len(label_ids):
                    allowed.add(eos_token_id)
                else:
                    # Next token in this label
                    allowed.add(label_ids[len(generated)])

        # If nothing matches, force EOS
        if not allowed:
            return [eos_token_id]

        return list(allowed)

    return allowed_tokens_fn



# Inference
def predict_labels(
    model,
    tokenizer,
    texts,
    labels,
    prompt_fn,
    max_new_tokens=10,
    verbose=True
):
    """ Predict labels using prefix-constrained decoding"""
    # Prefix-constrained decoding setup
    label_token_ids = build_label_token_map(labels, tokenizer)

    preds = []
    for i, text in enumerate(tqdm(texts, disable=not verbose, bar_format='{n_fmt}/{total_fmt}') if verbose else enumerate(texts)):
        if not verbose:
            i, text = text
        prompt = prompt_fn(text, labels)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        prompt_len = inputs["input_ids"].shape[-1]
        
        allowed_tokens_fn = make_allowed_tokens_fn(label_token_ids, eos_token_id=tokenizer.eos_token_id, prompt_len=prompt_len)
        logits_processor = [PrefixConstrainedLogitsProcessor(allowed_tokens_fn, num_beams=1)]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=1, # Reduces list-style answers
                do_sample=False, # Disable sampling for deterministic output
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=logits_processor,
                )

        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        #cleaned = normalize_label(decoded, labels)
        #preds.append(cleaned)
        preds.append(decoded)

        #if cleaned == "INVALID" and len(preds) <= 30 and verbose:
        #    print("PROMPT:\n", prompt)
        #    print("RAW OUTPUT:\n", decoded)
        #    print("CLEANED OUTPUT:\n", cleaned)
        #    print("-" * 80)

    return preds
