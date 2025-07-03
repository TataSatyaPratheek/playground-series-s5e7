# src/predict.py
import pandas as pd
import joblib
import yaml
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    logger.info("Starting prediction process")
    
    cfg = yaml.safe_load(open("src/config.yaml"))
    logger.info("Configuration loaded")
 
    logger.info(f"Loading model from {cfg['model']['path']}")
    model = joblib.load(cfg["model"]["path"])
    logger.info(f"Loading test data from {cfg['data']['test_path']}")
    test = pd.read_csv(cfg["data"]["test_path"])
    logger.info(f"Test data shape: {test.shape}")
    
    logger.info("Generating predictions...")
    pred = model.predict(test)
    logger.info("Note: Metrics cannot be calculated on test data as true labels are not available.")
    logger.info(f"Predictions generated. Distribution: {pd.Series(pred).value_counts().to_dict()}")


    sub = pd.read_csv(cfg["data"]["sample_path"])
    sub["Personality"] = pred
    sub.to_csv(cfg["submission"]["path"], index=False)
    print("Submission file written to", cfg["submission"]["path"])
    logger.info(f"Submission file written to {cfg['submission']['path']}")
    logger.info(f"Prediction process completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
