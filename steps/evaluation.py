import logging
import pandas as pd
from zenml import step
from src.evaluation_model import MAE,RMSE,R2
from sklearn.base import RegressorMixin
from typing import Annotated
from typing_extensions import Tuple
import mlflow
from zenml.client import Client
experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker="mlflow_tracker")

def evaluation_model(model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.DataFrame)->Tuple[
    Annotated[float,"mea"],
    Annotated[float,"r2"],
    Annotated[float,"rmse"],
]:
    try:
        prediction=model.predict(X_test)
        mea_class=MAE()
        mea=mea_class.calculate_score(y_test,prediction)
        mlflow.log_metric("mae",mea)

        r2_class=R2()
        r2=r2_class.calculate_score(y_test,prediction)
        mlflow.log_metric("r2",r2)

        rmse_class=RMSE()
        rmse=rmse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("rmse",rmse)

        return mea,r2,rmse
    except Exception as e:
        logging.error("Error")
        raise e
