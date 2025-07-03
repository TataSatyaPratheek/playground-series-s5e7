# src/train.py
import pandas as pd
import joblib
import yaml
import logging
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from .preprocessing import build_preprocess_pipeline
import numpy as np
from .evaluate import log_classification_metrics
from .feature_engineering import PersonalityFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


# Update preprocessing pipeline
def build_enhanced_pipeline(X, target_col):
    preprocessor = build_preprocess_pipeline(X, target_col)
    
    return Pipeline([
        ("feature_eng", PersonalityFeatureEngineer()),
        ("prep", preprocessor),
    ])


def train_model():
    start_time = time.time()
    logger.info("Starting model training process")
    cfg = yaml.safe_load(open("src/config.yaml"))
    logger.info(f"Configuration loaded: {cfg}")    
    df = pd.read_csv(cfg["data"]["train_path"])
    logger.info(f"Training data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    X, y = df.drop(columns=[cfg["data"]["target"]]), df[cfg["data"]["target"]]
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    # Build preprocessing pipeline
    preprocessor = build_preprocess_pipeline(X, cfg["data"]["target"])
    logger.info("Base CatBoost model configured")
     
    # Enhanced CatBoost with better hyperparameters based on search results[6][11][13]
    base_model = CatBoostClassifier(
        iterations=1000,  # Increased from 300[6][14]
        depth=6,
        learning_rate=0.05,  # Slightly reduced for better convergence[13][14]
        l2_leaf_reg=3,  # Added L2 regularization[11][13]
        random_strength=0.8,  # Added random strength[11]
        bagging_temperature=1.0,  # Added bagging temperature[11]
        border_count=128,  # Optimized border count[11]
        random_state=cfg["seed"],
        verbose=False,
    )
    
    # Create pipeline
    pipe = Pipeline([("prep", preprocessor), ("clf", base_model)])
    
    # Hyperparameter tuning with GridSearchCV[14]
    param_grid = {
        'prep__feature_select__k': [3, 5, 8],  # Feature selection tuning
        'clf__depth': [4, 6, 8],  # Tree depth tuning[14]
        'clf__learning_rate': [0.03, 0.05, 0.07],  # Learning rate tuning[14]
        'clf__l2_leaf_reg': [1, 3, 5],  # L2 regularization tuning[14]
    }
    
    logger.info(f"Grid search parameters: {param_grid}")
    # Split data for validation (needed for early stopping)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipe, 
        param_grid, 
        cv=3,  # Reduced CV folds to save memory on M1 Air
        scoring='accuracy',
        n_jobs=2,  # Limited jobs for M1 Air memory constraints
        verbose=1
    )
    
    logger.info("Starting grid search hyperparameter tuning...")
    # Fit with early stopping validation
    X_train_split = X.iloc[:int(0.8 * len(X))]
    X_val_split = X.iloc[int(0.8 * len(X)):]
    y_train_split = y.iloc[:int(0.8 * len(y))]
    y_val_split = y.iloc[int(0.8 * len(y)):]
    
    logger.info(f"Train split: {X_train_split.shape[0]} samples, Validation split: {X_val_split.shape[0]} samples")
    grid_search.fit(X_train_split, y_train_split)
    
    logger.info(f"Grid search completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    # Extract best parameters for creating new model
    best_params = grid_search.best_params_
    best_clf_params = {k.replace('clf__', ''): v for k, v in best_params.items() if k.startswith('clf__')}
    best_prep_params = {k: v for k, v in best_params.items() if not k.startswith('clf__')}    

    # Transform the validation set using the preprocessor from the best grid search estimator.
    # This is crucial because the eval_set bypasses the pipeline's transform steps.
    logger.info("Transforming validation data for eval_set...")
    X_val_transformed = grid_search.best_estimator_.named_steps['prep'].transform(X_val_split)
    logger.info("Validation data transformed.")
    
    best_pipe = grid_search.best_estimator_
    
    # Full cross-validation with best model
    scores = cross_val_score(best_pipe, X, y, cv=cv, scoring="accuracy", n_jobs=2)
    logger.info(f"Final CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Create NEW model with best parameters and early stopping for final training
    logger.info("Creating final model with best parameters and early stopping...")
    final_model = CatBoostClassifier(
        iterations=1000,
        depth=best_clf_params.get('depth', 6),
        learning_rate=best_clf_params.get('learning_rate', 0.05),
        l2_leaf_reg=best_clf_params.get('l2_leaf_reg', 3),
        random_strength=0.8,
        bagging_temperature=1.0,
        border_count=128,
        random_state=cfg["seed"],
        verbose=False,
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    # Create final pipeline with new model
    final_preprocessor = build_preprocess_pipeline(X, cfg["data"]["target"])
    # Extract and set the best feature selection parameter correctly[9]
    best_k = best_params.get('prep__feature_select__k', 8)
    final_preprocessor.set_params(feature_select__k=best_k)
    logger.info(f"Set feature selection k={best_k}")

    
    final_pipe = Pipeline([("prep", final_preprocessor), ("clf", final_model)])
    
    logger.info("Training final model with early stopping on full dataset...")
    # Fit the final pipeline on the full dataset (X, y).
    # The eval_set for early stopping MUST be pre-transformed to match the data format
    # that the classifier receives after the 'prep' step.
    final_pipe.fit(X, y, clf__eval_set=(X_val_transformed, y_val_split))
    logger.info(f"Final model training completed. Best iteration: {final_model.get_best_iteration() if hasattr(final_model, 'get_best_iteration') else 'N/A'}")

    # Evaluate the final model on the validation set
    logger.info("Evaluating final model on the hold-out validation set...")
    val_predictions = final_pipe.predict(X_val_split)
    val_probabilities = final_pipe.predict_proba(X_val_split)[:, 1] # Probability of the positive class ('Extrovert')
    
    log_classification_metrics(y_val_split, val_predictions, val_probabilities)
    
    # Save model
    joblib.dump(final_pipe, cfg["model"]["path"])
    logger.info(f"Model saved to {cfg['model']['path']}")
    logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    train_model()
