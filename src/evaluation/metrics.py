import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None, num_classes: int = 8) -> dict:
    """
    Computes full medical diagnostic metrics:
    - Overall Accuracy
    - Macro Sensitivity (Recall)
    - Macro Specificity
    - Macro Precision
    - Macro F1-Score
    - Multiclass Macro ROC-AUC
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    sens = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calculate specificity per class and take macro average
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        specificities.append(spec)
    macro_spec = float(np.mean(specificities))
    
    # AUC-ROC
    auc = np.nan
    if y_probs is not None:
        try:
            auc = float(roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro'))
        except Exception:
            pass

    return {
        "accuracy": round(float(acc), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(macro_spec), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "auc_roc": round(float(auc), 4) if not np.isnan(auc) else "N/A"
    }

def aggregate_5fold_results(fold_metrics_list: list) -> pd.DataFrame:
    """
    Computes Mean +/- Std Dev across all 5 folds.
    Returns clean summary DataFrame.
    """
    df = pd.DataFrame(fold_metrics_list)
    summary = {}
    for col in df.columns:
        if col != "fold":
            numeric_vals = pd.to_numeric(df[col], errors='coerce').dropna()
            mean_val = numeric_vals.mean()
            std_val = numeric_vals.std()
            summary[col] = f"{mean_val:.4f} +/- {std_val:.4f}"
            
    summary_df = pd.DataFrame(summary, index=["5-Fold Mean +/- Std"])
    return summary_df
