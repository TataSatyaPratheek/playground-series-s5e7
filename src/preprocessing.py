# src/preprocessing.py
import pandas as pd
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def build_preprocess_pipeline(df: pd.DataFrame, target_col: str = None):
    logger.debug("Building preprocessing pipeline")
    

    cat_cols = df.select_dtypes("object").columns.tolist()
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    
    # Remove target if present
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)
    logger.debug(f"Categorical columns: {cat_cols}")
    logger.debug(f"Numerical columns: {num_cols}")
    
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),  # Add scaling for numerical features
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )
    
    logger.debug("Preprocessing pipeline created successfully")
    # Add feature selection pipeline
    return Pipeline([
        ("preprocess", preprocessor),
        ("feature_select", SelectKBest(mutual_info_classif, k=8)),  # Will tune k
    ])
