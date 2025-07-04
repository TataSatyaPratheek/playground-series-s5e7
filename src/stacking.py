# src/stacking.py
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

def get_base_models(cfg):
    """Defines the base models for the ensemble."""
    # Using strong, pre-vetted defaults. These could be tuned per-model.
    catboost_params = cfg.get('catboost_params', {'depth': 4, 'learning_rate': 0.02, 'l2_leaf_reg': 4.5, 'iterations': 1000})
    lightgbm_params = cfg.get('lightgbm_params', {'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 20})
    xgb_params = cfg.get('xgb_params', {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 4, 'eval_metric': 'logloss', 'early_stopping_rounds': 100})

    models = {
        'catboost': CatBoostClassifier(**catboost_params, random_state=cfg['seed'], verbose=0),
        'lightgbm': LGBMClassifier(**lightgbm_params, random_state=cfg['seed']),
        'xgboost': XGBClassifier(**xgb_params, random_state=cfg['seed'])
    }
    return models

def generate_oof_predictions(base_models, feature_sets, y, cfg):
    """Generates OOF predictions using specified feature sets for each model."""
    logger.info("--- Generating Out-of-Fold (OOF) predictions for stacking ---")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg['seed'])
    
    oof_train = np.zeros((y.shape[0], len(base_models)))
    oof_test = np.zeros((feature_sets['eda'][1].shape[0], len(base_models))) # Use any test set for shape
    
    ensemble_config = cfg.get('ensemble_config', {})

    for i, (name, model) in enumerate(base_models.items()):
        feature_set_name = ensemble_config.get(name, {}).get('features', 'eda')
        X, X_test = feature_sets[feature_set_name]
        logger.info(f"Training base model: {name} using '{feature_set_name}' feature set.")
        
        test_preds_for_model = np.zeros((X_test.shape[0], n_splits))
        original_cat_features = [col for col in X.columns if X[col].dtype == 'object']
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
            X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

            # --- Model-Specific Fitting Logic ---
            if name == 'catboost':
                model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], cat_features=original_cat_features, early_stopping_rounds=100, verbose=False)
                positive_class_index = np.where(model.classes_ == 'Extrovert')[0][0]
            else:
                y_train_fold_encoded = y_train_fold.map({'Extrovert': 1, 'Introvert': 0})
                y_val_fold_encoded = y_val_fold.map({'Extrovert': 1, 'Introvert': 0})

                X_train_final = X_train_fold.copy()
                X_val_final = X_val_fold.copy()
                for col in original_cat_features:
                    X_train_final[col] = X_train_final[col].astype('category').cat.codes
                    X_val_final[col] = X_val_final[col].astype('category').cat.codes

                if name == 'lightgbm':
                    model.fit(X_train_final, y_train_fold_encoded, eval_set=[(X_val_final, y_val_fold_encoded)], callbacks=[lgb_early_stopping(100, verbose=False)])
                elif name == 'xgboost':
                    model.fit(X_train_final, y_train_fold_encoded, eval_set=[(X_val_final, y_val_fold_encoded)], verbose=False)
                
                positive_class_index = np.where(model.classes_ == 1)[0][0]

            # --- Predict Probabilities ---
            X_val_to_predict = X_val_final if name != 'catboost' else X_val_fold
            oof_train[val_idx, i] = model.predict_proba(X_val_to_predict)[:, positive_class_index]
            
            X_test_to_predict = X_test.copy()
            if name != 'catboost':
                 for col in original_cat_features:
                    X_test_to_predict[col] = X_test_to_predict[col].astype('category').cat.codes
            test_preds_for_model[:, fold] = model.predict_proba(X_test_to_predict)[:, positive_class_index]
            
        oof_test[:, i] = test_preds_for_model.mean(axis=1)

    return oof_train, oof_test
