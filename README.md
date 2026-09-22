# 🏡 End-to-End MLOps Pipeline: House Price Prediction

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline for predicting house prices. The project utilizes **ZenML** to orchestrate the machine learning workflows and **MLflow** for robust experiment tracking and model registry.

## 🚀 Project Overview

The goal of this project is to demonstrate a production-ready MLOps architecture using a classic House Price Prediction dataset. Instead of a standard Jupyter Notebook approach, this codebase is structured into modular, reusable steps—ensuring reproducibility, scalability, and seamless tracking of model metrics and artifacts.

## 🛠️ Tech Stack

* **Orchestration:** [ZenML](https://zenml.io/)
* **Experiment Tracking:** [MLflow](https://mlflow.org/)
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Language:** Python 3.x

## 📂 Repository Structure

The repository is organized to separate data, core logic, and pipeline steps:

```text
MLops-Project/
├── data/               # Raw and processed datasets
├── pipeline/           # ZenML pipeline definitions uniting the steps
├── src/                # Core logic and helper functions
├── steps/              # Individual ZenML pipeline steps
│   ├── clean_data.py   # Data preprocessing and feature engineering
│   ├── config.py       # Configuration and hyperparameter settings
│   ├── evaluation.py   # Model evaluation metrics (MSE, RMSE, R2 Score)
│   ├── ingest_data.py  # Loading the dataset from the source
│   └── model_train.py  # Training the regression model
├── .gitignore
├── __init__.py
└── run_pipeline.py     # Execution script to trigger the ZenML pipeline
