from pipeline.train_pipeline import training_pipeline
from zenml.client import Client
import mlflow
if __name__=="__main__":
    print(f"Active stack tracker: {Client().active_stack.experiment_tracker.name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    training_pipeline(data_path=r"data\Housing.csv")
