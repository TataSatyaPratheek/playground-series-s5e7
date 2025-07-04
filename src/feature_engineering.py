# src/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class EnhancedEDAFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates new features based on insights from the enhanced EDA:
    1. Composite 'Solitude_Preference_Score'.
    2. Composite 'Social_Engagement_Score'.
    3. The first two principal components (PC1, PC2) from all numeric data.
    """
    def __init__(self):
        self.numeric_cols_ = None
        self.global_pca_pipeline_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        logger.info("Fitting EnhancedEDAFeatureEngineer...")
        
        # Identify numeric columns for transformations
        self.numeric_cols_ = X.select_dtypes(include=np.number).columns.drop('id', errors='ignore').tolist()
        
        # Pipeline for the Global PCA components
        self.global_pca_pipeline_ = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2))
        ])
        self.global_pca_pipeline_.fit(X[self.numeric_cols_])

        # Define the final feature names this transformer will output
        self.feature_names_out_ = list(X.columns) + ['Solitude_Preference_Score', 'Social_Engagement_Score', 'GlobalPC1', 'GlobalPC2']
        return self

    def transform(self, X):
        logger.info("Transforming data with EnhancedEDAFeatureEngineer...")
        X_new = X.copy()
        
        # --- Create Composite Scores ---
        # Impute necessary columns temporarily to calculate scores
        imputer_median = SimpleImputer(strategy='median')
        X_new[self.numeric_cols_] = imputer_median.fit_transform(X_new[self.numeric_cols_])

        # 1. Solitude Preference Score (from EDA Cell 5)
        # This score increases with time spent alone and feeling drained after socializing
        drained_map = X_new['Drained_after_socializing'].map({'Yes': 1, 'No': -1}).fillna(0)
        X_new['Solitude_Preference_Score'] = X_new['Time_spent_Alone'] + drained_map

        # 2. Social Engagement Score (from EDA Cell 5)
        # This score increases with more social activity
        X_new['Social_Engagement_Score'] = X_new['Social_event_attendance'] + X_new['Friends_circle_size'] + X_new['Post_frequency']

        # --- Create Global PCA Features ---
        global_pcs = self.global_pca_pipeline_.transform(X[self.numeric_cols_]) # Use original X to avoid double imputation
        X_new['GlobalPC1'] = global_pcs[:, 0]
        X_new['GlobalPC2'] = global_pcs[:, 1]
        
        return X_new

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

