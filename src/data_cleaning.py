

import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union, Tuple
from sklearn.preprocessing import StandardScaler

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series, Tuple]:
        pass

class DataPreProcess(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

            # Convert 'yes' to 1 and 'no' to 0
            for col in binary_cols:
                data[col] = data[col].map({'yes': 1, 'no': 0})

            # One-hot encode 'furnishingstatus' (drops the first column to avoid multicollinearity)
            data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)
            return data

        except Exception as e:
             logging.error(f"Error : {e}")
             raise e
             
class DataDivide(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            X = data.drop(['price'], axis=1)
            y = data['price']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
            # Initialize the scaler
            scaler = StandardScaler().set_output(transform="pandas")

# Fit on training data and transform it
            X_train_scaled = scaler.fit_transform(X_train)

# Only transform the test data (do NOT fit, to prevent data leakage)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
             logging.error(f"Error : {e}")
             raise e

class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    # FIXED: This is now properly dedented so it is a class method
    def handle_data(self) -> Union[pd.DataFrame,pd.Series,Tuple]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error :{e}")
            raise e      

                    
             
        