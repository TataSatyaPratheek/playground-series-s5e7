# src/preprocessing.py
import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def build_advanced_preprocessor():
    """Builds a preprocessor for imputation and scaling, letting CatBoost handle encoding."""
    categorical_transformer = SimpleImputer(strategy='most_frequent')
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, selector(dtype_include=object)),
            ('num', numeric_transformer, selector(dtype_include=np.number))
        ],
        remainder='passthrough',
        verbose_feature_names_out=False # This gives clean feature names
    )
    preprocessor.set_output(transform="pandas") # CRITICAL: Ensure output is a DataFrame
    return preprocessor
