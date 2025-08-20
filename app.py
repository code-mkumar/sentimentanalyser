import streamlit as st
import tweepy
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources
nltk.download('vader_lexicon')

# ✅ Use Bearer Token (API v2)
bearer_token = "AAAAAAAAAAAAAAAAAAAAAE6r3gEAAAAAKKG3pNn%2B99FMO51dxLzastGByDQ%3DcIbcUX0YnD5NuEWBsmwf6mF0gAcawUsT1zQ5r5gMyd35WBSjxU"

# Authenticate with Twitter API v2
client = tweepy.Client(bearer_token=bearer_token)

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
st.title("📊 Twitter Sentiment Analysis App")

# User Input
topic = st.text_input("Enter a topic to analyze (e.g., COVID vaccine, AI, elections):")

if st.button("Analyze"):
    if topic.strip() == "":
        st.warning("⚠️ Please enter a topic!")
    else:
        query = f"{topic} -is:retweet lang:en"

        # Collect Tweets
        with st.spinner("Fetching tweets..."):
            tweets = client.search_recent_tweets(query=query,
                                                 max_results=100,
                                                 tweet_fields=["created_at", "text"])

        if not tweets.data:
            st.error("❌ No tweets found for this topic.")
        else:
            tweet_list = [[tweet.text, tweet.created_at] for tweet in tweets.data]
            df = pd.DataFrame(tweet_list, columns=['tweet', 'time'])

            st.success(f"✅ Collected {df.shape[0]} Tweets")
            st.dataframe(df.head())

            # Preprocessing
            def clean_tweet(text):
                text = re.sub(r"http\S+|www\S+", "", text)
                text = re.sub(r"@\w+|#\w+", "", text)
                text = re.sub(r"[^A-Za-z\s]", "", text)
                text = text.lower().strip()
                return text

            df['clean_tweet'] = df['tweet'].apply(clean_tweet)

            # Sentiment Analysis
            analyzer = SentimentIntensityAnalyzer()

            def get_sentiment(text):
                score = analyzer.polarity_scores(text)
                if score['compound'] >= 0.05:
                    return "Positive"
                elif score['compound'] <= -0.05:
                    return "Negative"
                else:
                    return "Neutral"

            df['Sentiment'] = df['clean_tweet'].apply(get_sentiment)

            # Results Summary
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df['Sentiment'].value_counts()
            st.write(sentiment_counts)

            # Pie Chart
            fig, ax = plt.subplots()
            sentiment_counts.plot.pie(autopct='%1.1f%%',
                                      colors=['green', 'red', 'gray'],
                                      figsize=(6,6),
                                      ax=ax)
            ax.set_title(f"Public Sentiment on '{topic}'")
            ax.set_ylabel("")
            st.pyplot(fig)

            # Word Clouds
            st.subheader("☁️ Word Clouds")
            positive_text = " ".join(df[df['Sentiment']=="Positive"]['clean_tweet'])
            negative_text = " ".join(df[df['Sentiment']=="Negative"]['clean_tweet'])

            if positive_text:
                wc_pos = WordCloud(width=600, height=400, background_color="white", colormap="Greens").generate(positive_text)
            else:
                wc_pos = WordCloud(width=600, height=400, background_color="white").generate("No Positive Tweets")

            if negative_text:
                wc_neg = WordCloud(width=600, height=400, background_color="white", colormap="Reds").generate(negative_text)
            else:
                wc_neg = WordCloud(width=600, height=400, background_color="white").generate("No Negative Tweets")

            col1, col2 = st.columns(2)

            with col1:
                st.image(wc_pos.to_array(), use_container_width=True, caption="Positive Tweets")

            with col2:
                st.image(wc_neg.to_array(), use_container_width=True, caption="Negative Tweets")

            # Show Data
            st.subheader("📑 Analyzed Tweets")
            st.dataframe(df[['time','tweet','Sentiment']])
