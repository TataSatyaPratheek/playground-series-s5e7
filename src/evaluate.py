# src/evaluate.py
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

def log_classification_metrics(y_true, y_pred, y_pred_proba, pos_label=1, target_names=None, report_path="reports"):
    """Calculates, logs, and saves a suite of classification metrics."""
    logger.info("--- Model Evaluation Metrics ---")
    
    # Calculate and log individual metrics
    accuracy = accuracy_score(y_true, y_pred)
    # Pass the correct numerical pos_label for calculation
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Use target_names for display, default to pos_label if not provided
    pos_label_name = target_names[pos_label] if target_names and len(target_names) > pos_label else pos_label

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision ({pos_label_name}): {precision:.4f}")
    logger.info(f"Recall ({pos_label_name}): {recall:.4f}")
    logger.info(f"F1-Score ({pos_label_name}): {f1:.4f}")
    logger.info(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Log detailed classification report with proper names
    logger.info("\n" + classification_report(y_true, y_pred, target_names=target_names))
    
    # Generate and save confusion matrix with proper labels
    cm_labels = target_names if target_names else np.unique(y_true)
    # Use actual unique values from y_true to define the matrix axes
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{report_path}/confusion_matrix.png")
    logger.info(f"Confusion matrix saved to {report_path}/confusion_matrix.png")
    logger.info("---------------------------------")
