import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import MyStepConfig
import mlflow
from zenml.client import Client

experiment_tracker=Client().active_stack.experiment_tracker


@step(experiment_tracker="mlflow_tracker")
def train_model(X_train:pd.DataFrame,X_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series)->RegressorMixin:

    try:
        model=None
        config=MyStepConfig()
        if config.model_name=="LinearRegression":
            mlflow.sklearn.autolog()
            model=LinearRegressionModel()
            trained_model=model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model is supported")
        
    except Exception as e:
        logging.error("Error:{}".format(e))
        raise e    