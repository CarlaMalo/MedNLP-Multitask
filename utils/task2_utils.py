"""
Utility functions for Task 2 (NER evaluation and metrics storage)
"""

import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support


def store_ner_metrics(y_true, y_pred, results_path, labels=None):
    """
    Compute and store NER model metrics (precision, recall, f1) in a CSV file.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        results_path: Path to save the results CSV file
        labels: Optional list of label names to restrict evaluation 
    """
    # Compute micro-averaged metrics
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    
    # Compute macro-averaged metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    
    # Create metrics dictionary
    metrics = {
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Save to CSV
    pd.DataFrame([metrics]).to_csv(results_path, index=False)
    
    return metrics


def print_ner_metrics(metrics):
    """
    Print NER metrics (easier to read)

    """
    # Micro metrics row
    micro_row = pd.DataFrame([{
        "Metric": "Micro",
        "Precision": f"{metrics['precision_micro']:.4f}",
        "Recall": f"{metrics['recall_micro']:.4f}",
        "F1": f"{metrics['f1_micro']:.4f}"
    }])
    
    # Macro metrics row
    macro_row = pd.DataFrame([{
        "Metric": "Macro",
        "Precision": f"{metrics['precision_macro']:.4f}",
        "Recall": f"{metrics['recall_macro']:.4f}",
        "F1": f"{metrics['f1_macro']:.4f}"
    }])
    
    # Combine and print
    result_df = pd.concat([micro_row, macro_row], ignore_index=True)
    print(result_df.to_string(index=False))
