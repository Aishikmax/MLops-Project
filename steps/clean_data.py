import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataPreProcess,DataDivide
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]

]:
    try:
        process_strategy=DataPreProcess()
        data_cleaning=DataCleaning(df,process_strategy)
        process_data=data_cleaning.handle_data()

        divide_strategy=DataDivide()
        data_cleaning=DataCleaning(process_data,divide_strategy)
        X_train,X_test,y_train,y_test=data_cleaning.handle_data()
        logging.info("Data cleaning complete")
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error("Error : {e}".format(e))
        raise e     


