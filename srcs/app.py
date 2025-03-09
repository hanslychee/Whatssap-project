import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Theme customization
st.set_page_config(page_title="Whatsapp Chat Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df, topics = preprocessor.preprocess(data)  # Get topics from preprocess
    file_name = uploaded_file.name.split('.')[0]
    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    # Add a selectbox for user selection
    selected_user = st.sidebar.selectbox("Show analysis for", user_list, index=0)

    # Display the overall analysis by default
    st.title("Overall Analysis")
    with st.expander("Show Analysis", expanded=True):
        # Perform analysis for the selected user
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        sns.lineplot(data=timeline, x='time', y='message', ax=ax, color='green')
        ax.set_title("Monthly Timeline")
        ax.set_xlabel("Time")
        ax.set_ylabel("Messages")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        sns.lineplot(data=daily_timeline, x='date', y='message', ax=ax, color='black')
        ax.set_title("Daily Timeline")
        ax.set_xlabel("Date")
        ax.set_ylabel("Messages")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            sns.barplot(x=busy_day.index, y=busy_day.values, ax=ax, color='purple')
            ax.set_title("Most Busy Day")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            sns.barplot(x=busy_month.index, y=busy_month.values, ax=ax, color='orange')
            ax.set_title("Most Busy Month")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Finding the busiest users in the group (Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                sns.barplot(x=x.index, y=x.values, ax=ax, color='red')
                ax.set_title("Most Busy Users")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        show_wordcloud = st.checkbox("Show Wordcloud")
        if show_wordcloud:
            st.title("Wordcloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

        # Most common words
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        sns.barplot(y=most_common_df[0], x=most_common_df[1], ax=ax)
        ax.set_title("Most Common Words")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
         # Convert month names to abbreviated format (e.g., "June" -> "Jun")
        # month_map = {
        #         'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
        #         'May': 'May', 'June': 'Jun', 'July': 'Jul', 'August': 'Aug',
        #         'September': 'Sep', 'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
        #     }
        # df['month'] = df['month'].map(month_map)

        #     # Group by month and sentiment
        # monthly_sentiment = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)

        #     # Plotting: Histogram (Bar Chart) for each sentiment
        # st.write("### Sentiment Count by Month (Histogram)")

        #     # Create a figure with subplots for each sentiment
        # fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        #     # Plot Positive Sentiment
        # axes[0].bar(monthly_sentiment.index, monthly_sentiment['positive'], color='green')
        # axes[0].set_title('Positive Sentiment')
        # axes[0].set_xlabel('Month')
        # axes[0].set_ylabel('Count')

        #     # Plot Neutral Sentiment
        # axes[1].bar(monthly_sentiment.index, monthly_sentiment['neutral'], color='blue')
        # axes[1].set_title('Neutral Sentiment')
        # axes[1].set_xlabel('Month')
        # axes[1].set_ylabel('Count')

        #     # Plot Negative Sentiment
        # axes[2].bar(monthly_sentiment.index, monthly_sentiment['negative'], color='red')
        # axes[2].set_title('Negative Sentiment')
        # axes[2].set_xlabel('Month')
        # axes[2].set_ylabel('Count')

            # Display the plots in Streamlit
        st.pyplot(fig)


        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            ax.set_title("Emoji Distribution")
            st.pyplot(fig)
            # Summary with sentences
        st.title(" Group Discussion Summary with Sentences")
        summary = helper.get_summary_with_sentences(df, topics, file_name)
        st.write(summary)
        # Topic Visualization
        st.title("Topic Visualization")
        for i, topic in enumerate(topics):
            st.subheader(f"Topic {i+1}")
            st.write(", ".join(topic))    

        # Plot the topic trends
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=helper.visualize_topic_trends(df, topics))
        plt.title('Topic Trends Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45)
        plt.legend(title='Topics', labels=[f"Topic {i+1}: {', '.join(topic)}" for i, topic in enumerate(topics)])
        plt.tight_layout()
        plt.show()
    # Extract and display example sentences
        example_sentences = helper.extract_example_sentences(df, topics)
        st.subheader("Example Sentences Based on Topics:")
        for topic, sentences in example_sentences.items():
            st.subheader(topic)
            for sentence in sentences:
                st.write("- " + sentence)

        