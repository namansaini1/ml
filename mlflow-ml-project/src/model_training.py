import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Assuming the dataset has 'abstract' and 'label' columns
    X = data['abstract']
    y = data['label']
    return X, y

def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', RandomForestClassifier())
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def main():
    mlflow.start_run()
    
    # Load and preprocess data
    data = load_data('data/arxiv_papers.csv')
    X, y = preprocess_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_text(report, "classification_report.txt")
    
    # Save the model
    joblib.dump(model, 'models/model.pkl')
    mlflow.sklearn.log_model(model, "model")
    
    mlflow.end_run()

if __name__ == "__main__":
    main()