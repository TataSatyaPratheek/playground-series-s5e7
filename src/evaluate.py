# src/evaluate.py
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

def log_classification_metrics(y_true, y_pred, y_pred_proba, report_path="reports"):
    """Calculates, logs, and saves a suite of classification metrics."""
    logger.info("--- Model Evaluation Metrics ---")
    
    # Calculate and log individual metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Extrovert')
    recall = recall_score(y_true, y_pred, pos_label='Extrovert')
    f1 = f1_score(y_true, y_pred, pos_label='Extrovert')
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision (Extrovert): {precision:.4f}")
    logger.info(f"Recall (Extrovert): {recall:.4f}")
    logger.info(f"F1-Score (Extrovert): {f1:.4f}")
    logger.info(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Log detailed classification report
    logger.info("\n" + classification_report(y_true, y_pred))
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['Introvert', 'Extrovert'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Introvert', 'Extrovert'], yticklabels=['Introvert', 'Extrovert'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{report_path}/confusion_matrix.png")
    logger.info(f"Confusion matrix saved to {report_path}/confusion_matrix.png")
    logger.info("---------------------------------")

