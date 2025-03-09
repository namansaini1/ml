import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_data(data):
    data['cleaned_abstract'] = data['abstract'].apply(clean_text)
    return data

def split_data(data, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_abstract'], data['label'], test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer