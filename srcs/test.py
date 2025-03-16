import re
import string
import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Configure logging for debugging and traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_text(text: str) -> str:
    """
    Preprocess the input text by lowering case, removing punctuation and numbers.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers using regex
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset and perform initial cleaning.
    Assumes a CSV file with columns 'text' and 'label' (binary sentiment).
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} records from {file_path}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Ensure necessary columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    
    # Clean the text column
    df['clean_text'] = df['text'].apply(clean_text)
    return df


def build_model(X_train: List[str], y_train: List[int]) -> (TfidfVectorizer, LogisticRegression):
    """
    Build a sentiment analysis pipeline consisting of a TF-IDF vectorizer and a Logistic Regression classifier.
    """
    # Initialize the TF-IDF vectorizer with stop word removal and n-gram consideration if needed
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    logging.info("TF-IDF vectorization complete.")

    # Initialize and train the Logistic Regression model
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    logging.info("Logistic Regression model training complete.")

    return vectorizer, clf


def evaluate_model(vectorizer: TfidfVectorizer, clf: LogisticRegression,
                   X_test: List[str], y_test: List[int]) -> None:
    """
    Evaluate the sentiment analysis model using the test dataset.
    """
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logging.info(f"Model Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)


def predict_sentiment(texts: List[str], vectorizer: TfidfVectorizer, clf: LogisticRegression) -> List[int]:
    """
    Predict sentiment labels for a list of texts.
    """
    texts_clean = [clean_text(text) for text in texts]
    texts_tfidf = vectorizer.transform(texts_clean)
    return clf.predict(texts_tfidf)


def main():
    # File path to your dataset; ensure your CSV file has 'text' and 'label' columns
    data_file = '/Users/caasidev/development/AI/last try/Whatssap-project/training_data/cleaned_chat_2.csv'
    
    # Load and clean data
    df = load_and_prepare_data(data_file)

    # Split the data into training and test sets (80/20 split)
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")

    # Build the model
    vectorizer, clf = build_model(X_train.tolist(), y_train.tolist())

    # Evaluate the model
    evaluate_model(vectorizer, clf, X_test.tolist(), y_test.tolist())

    # Example prediction
    sample_texts = [
        "I love this product! It works great and exceeded my expectations.",
        "Terrible experience. The service was slow and the quality was poor."
    ]
    predictions = predict_sentiment(sample_texts, vectorizer, clf)
    for text, label in zip(sample_texts, predictions):
        sentiment = 'Positive' if label == 1 else 'Negative'
        print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")


if __name__ == '__main__':
    main()
