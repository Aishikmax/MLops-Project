import logging 
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score


class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass


class MAE(Evaluation):
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            mea=mean_absolute_error(y_true,y_pred)
            logging.info("MEA={}".format(mea))
            return mea
        except Exception as e:
            logging.error("Error")
            raise e

class RMSE(Evaluation):
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            rmse=root_mean_squared_error(y_true,y_pred)
            logging.info("RMSE={}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error")
            raise e
        

class R2(Evaluation):
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            r2=r2_score(y_true,y_pred)
            logging.info("MEA={}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error")
            raise e        


