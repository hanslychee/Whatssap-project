import re
import pandas as pd
# from sentiment import predict_sentiment
# from translate import translate_text
import spacy
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOP_WORDS
from joblib import Parallel, delayed, parallel_backend
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator

translator = Translator()


# Load Spacy models once
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

# Merge English and French stop words
custom_stop_words = list(ENGLISH_STOP_WORDS.union(FRENCH_STOP_WORDS))

# Compile regex patterns once (optimized for performance)
media_pattern = re.compile(r"<media omitted>")
delete_pattern = re.compile(r"this message was deleted")
null_pattern = re.compile(r"null")
link_pattern = re.compile(r"http\S+|www\S+|https\S+")

def clean_message(text):
    """Remove media notifications, special characters, and unwanted symbols."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = media_pattern.sub("", text)  # Remove media notifications
    text = delete_pattern.sub("", text)
    text = null_pattern.sub("", text)
    text = link_pattern.sub("", text)  # Remove links
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)  # Remove special characters
    return text

def lemmatize_text(text, lang):
    """Lemmatize text based on the detected language."""
    nlp = nlp_fr if lang == 'fr' else nlp_en
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

def process_message(message):
    """Process a single message for lemmatization."""
    try:
        lang = detect(message)
        return lemmatize_text(message, lang)
    except LangDetectException:
        return ""

def preprocess(data):
    """Preprocess raw chat data into a structured DataFrame."""
    # Regex pattern to extract date, time, sender, and message
    pattern = r"^(?P<Date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<Time>[\d:]+(?:\S*\s?[AP]M)?)\s+-\s+(?:(?P<Sender>.*?):\s+)?(?P<Message>.*)$"

    filtered_messages = []
    valid_dates = []

    # Parse each line in the chat data
    for line in data.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            entry = match.groupdict()
            sender = entry.get("Sender")
            if sender and sender.strip().lower() != "system":  # Remove system messages
                filtered_messages.append(f"{sender.strip()}: {entry['Message']}")
                valid_dates.append(f"{entry['Date']}, {entry['Time'].replace('â€¯', ' ')}")

    # Create DataFrame
    df = pd.DataFrame({'user_message': filtered_messages, 'message_date': valid_dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p', errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Separate Users and Messages
    users, messages = [], []
    msg_pattern = r"^(.*?):\s(.*)$"
    for message in df["user_message"]:
        match = re.match(msg_pattern, message)
        if match:
            users.append(match.group(1))
            messages.append(match.group(2))
        else:
            users.append("group_notification")
            messages.append(message)

    df["user"] = users
    df["message"] = messages
    df = df[df["user"] != "group_notification"]
    df.reset_index(drop=True, inplace=True)

    # Store unfiltered messages
    df["unfiltered_messages"] = df["message"]

    # Clean messages
    df["message"] = df["message"].apply(clean_message)

    # Parallelize lemmatization using the threading backend
    with parallel_backend('threading'):
        df["lemmatized_message"] = Parallel(n_jobs=-1)(delayed(process_message)(msg) for msg in df["message"])
        # df['sentiment'] = Parallel(n_jobs=-1)(delayed(predict_sentiment)(msg) for msg in df["message"])

    # Drop original column
    df.drop(columns=["user_message"], inplace=True)

    # Extract time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    df['minute'] = df['date'].dt.minute

    # Sample data for topic modeling (reduce size for performance)
    sampled_df = df.sample(n=min(10000, len(df)), random_state=42)

    # Topic modeling
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=custom_stop_words)
    dtm = vectorizer.fit_transform(sampled_df['lemmatized_message'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)
    # translate the massages 
    translated_messages = []
    for message in sampled_df['message']:
        translated_message = translator.translate(message, dest='en')
        translated_messages.append(translated_message)
    df['translated_message'] = translated_messages
    # Assign topics to messages
    topic_results = lda.transform(dtm)
    sampled_df['topic'] = topic_results.argmax(axis=1)

    # Store topics for visualization
    topics = []
    for topic in lda.components_:
        topics.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print(topics)
    print(df["translated_message"])

    return df,topics
