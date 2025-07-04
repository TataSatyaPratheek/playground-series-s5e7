# src/train.py
import pandas as pd
import joblib
import yaml
import logging
import time
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Use your working relative imports
from .preprocessing import build_advanced_preprocessor
from .evaluate import log_classification_metrics
from .feature_engineering import EnhancedEDAFeatureEngineer
from .stacking import get_base_models, generate_oof_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def train_model():
    start_time = time.time()
    cfg = yaml.safe_load(open("src/config.yaml", "r"))
    
    # --- Load Data ---
    df = pd.read_csv(cfg["data"]["train_path"])
    X, y = df.drop(columns=[cfg["data"]["target"]]), df[cfg["data"]["target"]]
    X_test = pd.read_csv(cfg["data"]["test_path"])
    
    # =========================================================================
    # STAGE 1: PREPARE DIVERSE FEATURE SETS
    # =========================================================================
    logger.info("--- Stage 1: Preparing DIVERSE feature sets for the ensemble ---")

    # --- Feature Set 1: EDA-Driven (High-Level) ---
    eda_pipeline = Pipeline(steps=[
        ('feature_engineering', EnhancedEDAFeatureEngineer()),
        ('preprocessing', build_advanced_preprocessor())
    ])
    X_prepared_eda = eda_pipeline.fit_transform(X, y)
    X_test_prepared_eda = eda_pipeline.transform(X_test)
    logger.info(f"EDA feature set prepared with {X_prepared_eda.shape[1]} features.")

    # --- Feature Set 2: Raw (Low-Level) ---
    imputer_raw = SimpleImputer(strategy='median')
    X_prepared_raw = pd.DataFrame(imputer_raw.fit_transform(X.select_dtypes(include=np.number)), columns=X.select_dtypes(include=np.number).columns)
    X_test_prepared_raw = pd.DataFrame(imputer_raw.transform(X_test.select_dtypes(include=np.number)), columns=X_test.select_dtypes(include=np.number).columns)
    # Add back imputed categorical features
    for col in X.select_dtypes(include='object').columns:
        X_prepared_raw[col] = X[col].fillna(X[col].mode()[0])
        X_test_prepared_raw[col] = X_test[col].fillna(X_test[col].mode()[0])
    logger.info(f"Raw feature set prepared with {X_prepared_raw.shape[1]} features.")

    feature_sets = {
        'eda': (X_prepared_eda, X_test_prepared_eda),
        'raw': (X_prepared_raw, X_test_prepared_raw)
    }
    
    # =========================================================================
    # STAGE 2: STACKING ENSEMBLE
    # =========================================================================
    base_models = get_base_models(cfg)
    oof_train, oof_test = generate_oof_predictions(base_models, feature_sets, y, cfg)

    # --- Level 1: Train the Meta-Model ---
    logger.info("--- Training Level 1 Meta-Model on OOF predictions ---")
    meta_model = LogisticRegression(C=0.1, random_state=cfg['seed'])
    meta_model.fit(oof_train, y)
    
    os.makedirs(os.path.dirname(cfg["model"]["path"]), exist_ok=True)
    joblib.dump(meta_model, cfg["model"]["path"])
    logger.info(f"Meta-model saved to {cfg['model']['path']}")
    
    # --- Final Evaluation & Submission ---
    logger.info("--- Final Stacked Ensemble Evaluation ---")
    positive_class_index = np.where(meta_model.classes_ == 'Extrovert')[0][0]
    meta_oof_preds_proba = meta_model.predict_proba(oof_train)[:, positive_class_index]
    y_binary = y.map({'Extrovert': 1, 'Introvert': 0})
    log_classification_metrics(y_true=y_binary, y_pred=(meta_oof_preds_proba > 0.5).astype(int), y_pred_proba=meta_oof_preds_proba, target_names=['Introvert', 'Extrovert'])
    
    final_test_preds = meta_model.predict(oof_test)
    
    submission_df = pd.read_csv(cfg["data"]["sample_path"])
    submission_df[cfg["data"]["target"]] = final_test_preds
    submission_df.to_csv(cfg["submission"]["path"], index=False)
    
    logger.info(f"Stacked ensemble submission file created at {cfg['submission']['path']}")
    logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    train_model()
