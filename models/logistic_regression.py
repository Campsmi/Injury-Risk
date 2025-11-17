import logging
import datetime
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

logging.basicConfig(level=logging.DEBUG)


class ACL_Risk_Predictor:

    def __init__(self, args):
        self.args = args
    
    def train(self, X_train, y_train):
        logging.info("Constructing model...")

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=500
        )

        logging.info("Fitting model...")

        model.fit(X_train, y_train)

        logging.info("Model successfully fit.")

        return model

    def save(self, model):
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_path = f"{self.args['output_path']}/model-{time}.pkl"

        joblib.dump(model, out_path)

        logging.info(f"Model saved to {out_path}.")







