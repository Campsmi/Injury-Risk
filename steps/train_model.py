import logging
import yaml
import numpy as np

import mlflow

from models.logistic_regression import ACL_Risk_Predictor

logging.basicConfig(level=logging.DEBUG)

def train_model(X_train, y_train):

    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    acl_risk_predictor = ACL_Risk_Predictor(configs)

    logging.info("Starting training...")
    model = acl_risk_predictor.train(X_train, y_train)

    logging.info("Saving model...")
    acl_risk_predictor.save(model)
    logging.info("Done.")
    return model