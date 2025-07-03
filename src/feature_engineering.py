# src/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class PersonalityFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering for personality detection[16]"""
    
    def __init__(self):
        logger.debug("PersonalityFeatureEngineer initialized")
        pass
    
    def fit(self, X, y=None):
        logger.debug("Fitting PersonalityFeatureEngineer")
        return self
    
    def transform(self, X):
        logger.debug(f"Transforming data with shape: {X.shape}")
        X_new = X.copy()
        
        # Interaction features based on domain knowledge[12]
        if 'Age' in X_new.columns and 'Post_frequency' in X_new.columns:
            X_new['Age_Post_interaction'] = X_new['Age'] * X_new['Post_frequency']
            logger.debug("Added Age_Post_interaction feature")

        # Binning continuous features[12]
        if 'Age' in X_new.columns:
            X_new['Age_group'] = pd.cut(X_new['Age'], bins=3, labels=['Young', 'Middle', 'Senior'])
            logger.debug("Added Age_group binned feature")

        # Ratio features[12]
        if 'Post_frequency' in X_new.columns and 'Following' in X_new.columns:
            X_new['Post_to_Following_ratio'] = X_new['Post_frequency'] / (X_new['Following'] + 1)
            logger.debug("Added Post_to_Following_ratio feature")
        
        return X_new
