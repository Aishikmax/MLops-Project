from pydantic import BaseModel

class MyStepConfig(BaseModel):
    model_name:str="LinearRegression"
