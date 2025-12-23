from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np

def store_model_metrics(y_true, y_pred, unique_labels, results_path):
    if results_path is None:
        return "Error: specify `results_path`"
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=sorted(unique_labels),
        zero_division=np.nan
    )   

    per_label_df = pd.DataFrame({
        "label": sorted(unique_labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    })
    per_label_df.to_csv(results_path, index=False)
    print("Metrics saved to ", results_path)