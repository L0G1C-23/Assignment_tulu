import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

def load_dataset():
    """Load and validate the dataset"""
    try:
        df = pd.read_csv('data/messages.csv')
        return df
    except FileNotFoundError:
        print("Error: data/messages.csv not found!")
        return None

def preprocess_data(df):
    """Basic data preprocessing"""
    # Remove any duplicates
    df = df.drop_duplicates()
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: Found missing values, removing them...")
        df = df.dropna()
    
    return df

def train_model(df):
    """Train the ML model with proper evaluation"""
    
    # Split the data
    X = df['text']
    y = df['label']
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the model pipeline
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # unigrams and bigrams
        max_features=5000,   # limit vocabulary size
        stop_words='english',
        sublinear_tf=True,      # better scaling
        smooth_idf=True,         # smoother weights
        lowercase=True,
        strip_accents='ascii'
    )
    
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    
    labels = ['appointment', 'billing', 'reports', 'complaint']
    for label in labels:
        if label in report:
            f1 = report[label]['f1-score']
            print(f"{label.capitalize()} F1: {f1:.3f}")
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1: {macro_f1:.3f}")
    
    ''' ===USE FOR CONFUSION MATRIX===
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Predicted ->", end="")
    for label in labels:
        print(f"{label[:8]:>10}", end="")
    print()
    
    for i, true_label in enumerate(labels):
        print(f"Actual {true_label[:8]:<8}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>10}", end="")
        print()    
    '''

    return pipeline


def save_model_artifacts(pipeline):
    """Save the trained model and vectorizer"""
    try:
        # Extract components from pipeline
        trained_vectorizer = pipeline.named_steps['tfidf']
        trained_classifier = pipeline.named_steps['classifier']
        
        # Save the vectorizer and model separately for easier loading
        joblib.dump(trained_vectorizer, 'models/vectorizer.joblib')
        joblib.dump(trained_classifier, 'models/model.joblib')
        
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    create_directories()
    
    df = load_dataset()
    if df is None:
        return
    
    if len(df) < 100:
        print(f"Warning: Dataset has only {len(df)} rows. Minimum requirement is 100.")
    
    df = preprocess_data(df)
    
    pipeline = train_model(df)
    
    save_model_artifacts(pipeline)

if __name__ == "__main__":
    main()