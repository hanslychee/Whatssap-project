from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'percentage', 'user': 'Name'})
    return x,df

def create_wordcloud(selected_user, df):
    # f = open('stop_hinglish.txt', 'r')
    stop_words = df

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.lower().str.contains('<media omitted>')]

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    # f = open('stop_hinglish.txt','r')
    stop_words = df

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.lower().str.contains('<media omitted>')]

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['unfiltered_messages']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year','month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap





def visualize_topic_trends(df, topics):
    # Create a DataFrame to store topic trends over time
    topic_trends = pd.DataFrame()

    # Count the number of messages for each topic per month
    for i, topic in enumerate(topics):
        topic_name = f"Topic {i+1}"
        df[topic_name] = df['lemmatized_message'].apply(lambda x: any(word in x for word in topic))
        monthly_counts = df.resample('M', on='date')[topic_name].sum()
        topic_trends[topic_name] = monthly_counts
        return topic_trends

def get_top_topics(df, topics, file_name):
    # Calculate the frequency of each topic in the chat
    topic_frequencies = []
    for i, topic in enumerate(topics):
        topic_name = f"Topic {i+1}"
        df[topic_name] = df['lemmatized_message'].apply(lambda x: any(word in x for word in topic))
        topic_count = df[topic_name].sum()
        topic_frequencies.append((topic_name, topic_count, topic))

    # Sort topics by frequency
    topic_frequencies.sort(key=lambda x: x[1], reverse=True)

    # Get the top 3 topics
    top_topics = topic_frequencies[:3]

    # Generate summary statement
    top_topics_summary = ", ".join([f"{name} ({', '.join(words[:3])})" for name, _, words in top_topics])
    summary_statement = f"In this Group ({file_name}), this is their main focus of discussion: {top_topics_summary}."

    return summary_statement

def construct_sentences_from_topics(topics):
    # Define a template for constructing sentences
    sentence_templates = [
        "The group frequently discusses topics related to {}.",
        "A major focus of the group's discussions is {}.",
        "The members often talk about {}."
    ]

    # Construct sentences for each topic
    sentences = []
    for i, topic in enumerate(topics):
        # Join the top words in the topic to form a context
        context = ", ".join(topic[:3])  # Use the top 3 words for context
        sentence = sentence_templates[i % len(sentence_templates)].format(context)
        sentences.append(sentence)

    return sentences

def get_summary_with_sentences(df, topics, file_name):
    # Construct sentences from topics
    sentences = construct_sentences_from_topics(topics)

    # Generate summary statement
    summary_statement = f"In this Group ({file_name}), here are the main focuses of discussion:\n" + "\n".join(sentences)

    return summary_statement

def extract_example_sentences(df, topics, num_sentences=3):
    # Store example sentences for each topic
    topic_sentences = {}

    for i, topic in enumerate(topics):
        topic_name = f"Topic {i+1}"
        # Filter messages that contain any of the top words in the topic
        topic_messages = df[df['lemmatized_message'].apply(lambda x: any(word in x for word in topic))]

        # Select a few example sentences from these messages
        example_sentences = topic_messages['unfiltered_messages'].head(num_sentences).tolist()
        topic_sentences[topic_name] = example_sentences

    return topic_sentences















