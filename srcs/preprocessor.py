import re
import pandas as pd
import swifter
from sentiment import process_dataframe_sentiment


# Function to clean WhatsApp messages
def clean_whatsapp_message(text):
    if isinstance(text, float) and pd.isna(text):  # Check if it's NaN
        return ""  # Remove empty messages
    
    text = str(text)  # Convert to string
    text = re.sub(r'this message was deleted', '',   text) 
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<media omitted>', '', text, flags=re.IGNORECASE)  # Remove media placeholders
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)  # Remove emojis & special characters (keep letters)
    text = text.lower().strip()  # Convert to lowercase & remove extra spaces
    
    return text if len(text) > 1 else ""

def preprocess(data):
    pattern = r"^(?P<Date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<Time>[\d:]+(?:\S*\s?[AP]M)?)\s+-\s+(?:(?P<Sender>.*?):\s+)?(?P<Message>.*)$"

    attern = r"^(?P<Date>\d{1,2}/\d{1,2}/\d{2,4}),\s+[\d:]+(?:\S*\s?[AP]M)?\s+-\s+(?P<Sender>[^:]+):\s+(?P<Message>.*)$"

    filtered_messages = []

    for line in data.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            entry = match.groupdict()
            sender = entry.get("Sender")  # Get sender, could be None
            if sender and sender.strip().lower() != "system":  # Check if sender is not None
                filtered_messages.append(f"{sender.strip()}: {entry['Message']}")

    # filtered_messages

    pattern = r"^(?P<Date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<Time>[\d:]+(?:\S*\s?[AP]M)?)\s+-\s+(?:(?P<Sender>.*?):\s+)?(?P<Message>.*)$"
    # removing the system dates
    valid_dates = []

    for line in data.strip().split('\n'):
        match = re.match(pattern, line)
        if match:
            entry = match.groupdict()
            # If there's a sender, we keep the date
            if entry["Sender"] is not None:
                valid_dates.append(f"{entry['Date']}, {entry['Time'].replace('â€¯', ' ')}")
    # Create the DataFrame from  the filtered_messages  and valid_dates
    df = pd.DataFrame({'user_message': filtered_messages, 'message_date': valid_dates})

    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p')

    # Rename the 'message_date' column to 'date'
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Separate Users and Messages
    users = []
    messages = []
    pattern = r"^(.*?):\s(.*)$"  # Matches "User: Message"

    for message in df["user_message"]:
        match = re.match(pattern, message)
        if match:
            users.append(match.group(1))  # Extract username
            messages.append(match.group(2))  # Extract message
        else:
            users.append("group_notification")
            messages.append(message)

    df["user"] = users
    df["message"] = messages
    # Convert text to lowercase
    df['message'] = df['message'].str.lower()

    # Convert 'message' column to string data type
    df['message'] = df['message'].astype(str)
    # Drop original column
    df.drop(columns=["user_message"], inplace=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['message'] = df['message'].apply(clean_whatsapp_message) 
     # Apply cleaning properly
    #  drop all row that have null in the message column
    df = df.dropna(subset=['message'])

    new_df = df[['message']]

    processed_df = process_dataframe_sentiment(new_df[['message']])   
    print(processed_df)
    # remove the rows that the language is unknown
    processed_df = processed_df[processed_df['language'] != 'unknown']
    print(processed_df)
    return df
