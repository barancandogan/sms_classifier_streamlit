import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle

def train_spam_model():
    # Load the dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Assuming the first column is the label (spam/ham) and second column is the message
    X = df.iloc[:, 1].values  # Messages
    y = df.iloc[:, 0].values  # Labels
    
    # Convert labels to binary (0 for ham, 1 for spam)
    y = np.where(y == 'spam', 1, 0)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('svm', SVC(kernel='linear', probability=True))
    ])
    
    # Train the model
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Save the trained pipeline
    print("\nSaving the model...")
    with open('sms_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved as 'sms_pipeline.pkl'")

if __name__ == "__main__":
    train_spam_model() 