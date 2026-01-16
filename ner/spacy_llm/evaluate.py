import json
import spacy
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from utils.task2_utils import print_ner_metrics, store_ner_metrics
from ..utils import extract_spans, get_label_lists, remove_overlapping_spans
import time
import argparse
import warnings
from transformers import logging as transformers_logging

# Suppress transformers warnings about attention mask and pad token
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad token.*")

parser = argparse.ArgumentParser(description="Evaluate spaCy-LLM model")
# Models available: mistral-7b, dolly-v2-3b, LLama-2-7b-hf, Llama-2-13b-hf
parser.add_argument("--model", type=str, default="mistral-7b", help="Model name to use (default: mistral_7b)")
args = parser.parse_args()
model = args.model

script_dir = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(script_dir, "../samples/")
test_path = os.path.join(PATH, "annotated_samples_test_1.json")

# Load test annotations 
with open(test_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    sentences = [text for text, _ in data["annotations"]]
    gold_spans = [ann["entities"] for _, ann in data["annotations"]]

# Remove overlapping spans from gold annotations
gold_spans = remove_overlapping_spans(gold_spans)
print(f"Loaded {len(sentences)} annotated test sentences.")

# Load spaCy-LLM model 
print(f"Loading model...{model}")
model_path = os.path.join(script_dir, f"models/output_{model}_ner")
nlp = spacy.load(model_path)

# Measure run-time
start_time = time.time()

pred_spans = extract_spans(nlp, sentences)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Prediction completed in {elapsed_time:.2f} seconds for {len(sentences)} sentences.")

# Prepare label lists 
true_labels, pred_labels, associated_sentence_idx = get_label_lists(gold_spans, pred_spans)
 
labels = sorted(set(true_labels) | set(pred_labels))
entity_labels = [lab for lab in labels if lab != "NONE"]
entity_indices = [labels.index(lab) for lab in entity_labels]

# 5. Evaluate macro and micro F1 and save metrics to CSV
path_spacy_llm_ner = f"Task2/results/spacy_llm_{model}_ner.csv"
metrics = store_ner_metrics(true_labels, pred_labels, path_spacy_llm_ner, labels=entity_labels)

print("\n" + "=" * 80)
print("spaCy-LLM NER Evaluation")
print("=" * 80)
print_ner_metrics(metrics)

# Build confusion matrix over all labels
cm = confusion_matrix(true_labels, pred_labels, labels=labels)
# Per-label TP, FP, FN 
tp_per_label = {}
fp_per_label = {}
fn_per_label = {}
for i, lab in enumerate(labels):
    tp_l = cm[i, i]
    fn_l = cm[i, :].sum() - tp_l
    fp_l = cm[:, i].sum() - tp_l
    tp_per_label[lab] = int(tp_l)
    fp_per_label[lab] = int(fp_l)
    fn_per_label[lab] = int(fn_l)

print("\n==== Per-label Counts ====")
for lab in labels:
    print(f"{lab}: TP={tp_per_label[lab]}, FP={fp_per_label[lab]}, FN={fn_per_label[lab]}")

tp_entities = sum(cm[i, i] for i in entity_indices)
fp_entities = sum(cm[:, i].sum() - cm[i, i] for i in entity_indices)
fn_entities = sum(cm[i, :].sum() - cm[i, i] for i in entity_indices)

print(f"TP_entities: {tp_entities}, FP_entities: {fp_entities}, FN_entities: {fn_entities}")

# 6. Show some error cases
print("\n===== Examples of WRONG predictions =====\n")

for sent_id, (g, p) in enumerate(zip(gold_spans, pred_spans)):
    if g != p:
        print("Text:", sentences[sent_id])
        print("Gold:", g)
        print("Pred:", p)
        print("-" * 50)
        break
#7. Show some correct cases
print("\n===== Examples of CORRECT predictions =====\n")
for sent_id, (g, p) in enumerate(zip(gold_spans, pred_spans)):
    if g == p:
        print("Text:", sentences[sent_id])
        print("Gold:", g)
        print("Pred:", p)
        print("-" * 50)
        break