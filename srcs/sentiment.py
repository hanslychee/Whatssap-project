import re
import torch
import pandas as pd
import swifter
from langdetect import detect
from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


data = {
    "message": [
        "I love this product! It's amazing.",
        "No me gusta este producto. Es terrible.",
        "Ce produit est génial!",
        "Dies ist das schlechteste Produkt, das ich je gekauft habe.",
        "Questo è un prodotto fantastico!",
        "这个产品太糟糕了。",
        "この製品は素晴らしいです！",
        "이 제품은 정말 좋아요!",
        "Ce film était incroyable!",
        "Je déteste ce film."
    ]
}

df = pd.DataFrame(data)

# Load sentiment model with GPU support
def load_sentiment_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device)
    return sentiment_pipeline

model = load_sentiment_model()

# Clean text function (optimized with swifter)
def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text.lower().strip()  # Convert to lowercase and remove extra spaces

# Detect language function
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

# Map star ratings to Positive, Neutral, Negative
def map_sentiment(star_label):
    return {
        "1 star": "Negative",
        "2 stars": "Negative",
        "3 stars": "Neutral",
        "4 stars": "Positive",
        "5 stars": "Positive",
    }.get(star_label, "Unknown")

# Analyze sentiment in batches
def analyze_sentiment_batch(texts):
    results = model(texts)  # Process in batch
    return [(map_sentiment(res['label']), res['score']) for res in results]

def truncate_message(text, max_length=512):
    tokens = tokenizer.tokenize(text)  # Tokenize text
    if len(tokens) > max_length:
        text = tokenizer.convert_tokens_to_string(tokens[:max_length])  # Truncate to max tokens
    return text

# Function to process DataFrame efficiently
def process_dataframe_sentiment(df, batch_size=32):
    df = df.copy()  # Avoid modifying original df
    
    # Clean text and detect language in parallel
    df['message'] = df['message'].apply(truncate_message)
    df['cleaned_text'] = df['message'].swifter.apply(clean_text)
    df['language'] = df['message'].swifter.apply(detect_language)

    # Perform sentiment analysis in batches
    sentiments = []
    for i in range(0, len(df), batch_size):
        batch_texts = df['cleaned_text'].iloc[i:i+batch_size].tolist()
        batch_results = analyze_sentiment_batch(batch_texts)
        sentiments.extend(batch_results)

    df['Sentiment'], df['Confidence'] = zip(*sentiments)
    return df

