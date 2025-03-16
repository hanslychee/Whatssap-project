import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset as HFDataset

# Load all cleaned WhatsApp datasets
files = [
    "/Users/caasidev/development/AI/last try/Whatssap-project/training_data/cleaned_chat_2.csv", 
    "/Users/caasidev/development/AI/last try/Whatssap-project/training_data/cleaned_chat_3.csv",
    "/Users/caasidev/development/AI/last try/Whatssap-project/training_data/cleaned_chat_4.csv",
    "/Users/caasidev/development/AI/last try/Whatssap-project/training_data/cleaned_chat_5.csv",
    "/Users/caasidev/development/AI/last try/Whatssap-project/training_data/cleaned_chat_6.csv",
    "/Users/caasidev/development/AI/last try/Whatssap-project/training_data/final_cleaned_chat.csv",
]

# Merge all datasets
df_list = [pd.read_csv(file) for file in files]
df = pd.concat(df_list, ignore_index=True)

# Load multilingual tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)  # 3 labels: Negative, Neutral, Positive

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(list(map(str, examples["text"])), padding="max_length", truncation=True)

dataset = HFDataset.from_pandas(df)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert dataset to PyTorch format
train_dataset = tokenized_datasets.train_test_split(test_size=0.1)["train"].with_format("torch")
test_dataset = tokenized_datasets.train_test_split(test_size=0.1)["test"].with_format("torch")

# Define DataLoader
BATCH_SIZE = 16
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training and evaluation functions
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        labels = batch["label"].to(device)
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            labels = batch["label"].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss = evaluate(model, test_dataloader, criterion)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

torch.save(model.state_dict(), "multilingual_whatsapp_sentiment.pth")

# Sentiment Prediction
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).to(device)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    sentiment = ["Negative", "Neutral", "Positive"]
    return sentiment[predicted_class]

print(predict_sentiment("C'est une super journée!"))  # French
print(predict_sentiment("Это ужасно!"))  # Russian
print(predict_sentiment("J'ai besoin d'aide"))  # French (Neutral)
