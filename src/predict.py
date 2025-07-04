# src/predict.py
import pandas as pd
import joblib
import yaml
import logging
import time
import os
import glob
import numpy as np

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Generates predictions by averaging an ensemble of saved models."""
    start_time = time.time()
    cfg = yaml.safe_load(open("src/config.yaml"))
    logger.info("Starting ENSEMBLE prediction process")

    # --- Load Data ---
    test_df = pd.read_csv(cfg["data"]["test_path"])
    logger.info(f"Test data loaded with shape: {test_df.shape}")

    # --- Load Ensemble Models ---
    models_dir = cfg["model"].get("models_dir", "models/ensemble")
    model_paths = glob.glob(os.path.join(models_dir, "model_fold_*.joblib"))
    
    if not model_paths:
        logger.error(f"No models found in {models_dir}. Please run the training script first.")
        return

    logger.info(f"Found {len(model_paths)} models for ensembling.")
    
    # --- Generate & Average Predictions ---
    all_preds = []
    for path in model_paths:
        logger.debug(f"Loading model: {path}")
        model = joblib.load(path)
        # Predict probabilities; we want the probability of the positive class ('Extrovert')
        preds = model.predict_proba(test_df)[:, 1]
        all_preds.append(preds)
    
    # Average the probabilities across all models
    avg_preds = np.mean(all_preds, axis=0)
    logger.info("Averaged predictions from all models.")

    # Convert averaged probabilities to final labels
    final_labels = ["Extrovert" if p > 0.5 else "Introvert" for p in avg_preds]
    
    # --- Create Submission File ---
    submission_df = pd.read_csv(cfg["data"]["sample_path"])
    submission_df[cfg["data"]["target"]] = final_labels
    submission_df.to_csv(cfg["submission"]["path"], index=False)
    
    logger.info(f"Submission file written to {cfg['submission']['path']}")
    logger.info(f"Prediction process completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

