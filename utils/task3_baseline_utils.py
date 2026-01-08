from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import json
from collections import defaultdict

def store_model_metrics_manual(y_true, y_pred, results_path):
    """
    Store overall model metrics (accuracy, precision, recall, f1) in a CSV file
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        results_path: Path to save the results CSV file
    """
    metrics = {}

    metrics["f1_weighted"] = f1_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )
    
    metrics["f1_macro"] = f1_score(
    y_true,
    y_pred,
    average='macro',
    zero_division=0
)

    pd.DataFrame([metrics]).to_csv(results_path, index=False)

    

# TODO the following is not calculating values properly - all zeros
# def store_model_metrics(y_true, y_pred, unique_labels, results_path):
#     if results_path is None:
#         return "Error: specify `results_path`"
    
#     unique_labels = list(sorted(unique_labels))

#     precision, recall, f1, support = precision_recall_fscore_support(
#         y_true,
#         y_pred,
#         labels=unique_labels,
#         zero_division=0
#     )   

#     per_label_df = pd.DataFrame({
#         "label": unique_labels,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "support": support
#     })
#     per_label_df.to_csv(results_path, index=False)
#     print("Metrics saved to ", results_path)



def store_misclassified_examples(test_texts, test_labels, preds, output_path, max_per_label=5):
    """
    Store misclassified examples per label.
    
    Args:
        test_texts: List of test text samples
        test_labels: List of true labels
        preds: List of predicted labels
        output_path: Path to save the misclassified examples JSON file
        max_per_label: Maximum number of examples to store per label (default: 5)
    """
    misclassified = defaultdict(list)
    
    for text, true_label, pred_label in zip(test_texts, test_labels, preds):
        if true_label != pred_label:
            # Store examples per true label
            if len(misclassified[true_label]) < max_per_label:
                misclassified[true_label].append({
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": pred_label
                })
    
    # Convert defaultdict to regular dict for JSON serialization
    misclassified_dict = {
        label: examples
        for label, examples in sorted(misclassified.items())
    }
    
    with open(output_path, 'w') as f:
        json.dump(misclassified_dict, f, indent=2)
    
    # Print summary
    print(f"\nMisclassified examples saved to {output_path}")
    for label, examples in misclassified_dict.items():
        print(f"  {label}: {len(examples)} example(s)")