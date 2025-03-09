import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import joblib

def load_model(model_path):
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def log_evaluation_metrics(metrics):
    mlflow.log_metrics(metrics)

if __name__ == "__main__":
    model_path = "../models/model.pkl"
    test_data_path = "../data/arxiv_papers.csv"  # Adjust as necessary
    test_data = pd.read_csv(test_data_path)
    
    # Assuming the last column is the target variable
    X_test = test_data.iloc[:, :-1]  # Features
    y_test = test_data.iloc[:, -1]   # Target variable
    
    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test)
    log_evaluation_metrics(metrics)