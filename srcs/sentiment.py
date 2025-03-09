# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
# from transformers import pipeline


# MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# ROBERTA_SUPPORTED_LANGUAGES = ('ar', 'en', 'fr', 'de', 'hi', 'it', 'es', 'pt')

# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# config = AutoConfig.from_pretrained(MODEL)

# #/ save the model locally
# model.save_pretrained(MODEL)
# tokenizer.save_pretrained(MODEL)
# def preprocess(text):
#     remove_words = ["<Media", "<Mediaomitted>"]
#     # Remove unwanted words
#     for word in remove_words:
#         text = text.replace(word, "")
    
#     # Process each word in the text
#     new_text = [
#         '@user' if t.startswith('@') and len(t) > 1 else
#         'http' if t.startswith('http') else
#         t
#         for t in text.split(" ")
#     ]
#     return " ".join(new_text)


# def predict_sentiment(text: str) -> str:
#     processed_text = preprocess(text)
#     encoded_input = tokenizer(processed_text, return_tensors='pt')
#     output = model(**encoded_input)
#     index_of_sentiment = output.logits.argmax().item()
#     sentiment = config.id2label[index_of_sentiment]
#     return sentiment


# def detect_emotion(text):
#     """
#     Detects the emotion of a text using Hugging Face's emotion model.
#     """
    
#     emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", device=0)
#     result = emotion_classifier(text)
#     return result[0]['label']


# # text = "la pizza da @michele è veramente buona https://www.youtube.com"
# # # text = "این غذا خیلی شوره!"
# # # text = "یه جلسه دیگه که میتونست یه ایمیل باشه 🥲"
# # print(predict_sentiment(text))