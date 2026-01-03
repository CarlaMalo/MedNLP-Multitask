from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

def store_model_metrics_manual(y_true, y_pred, results_path):
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