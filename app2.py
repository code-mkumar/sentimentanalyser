import tweepy
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources (first time only)
nltk.download('vader_lexicon')

# ✅ Use Bearer Token (API v2)
bearer_token = "AAAAAAAAAAAAAAAAAAAAAF%2Bw3gEAAAAAhHGfewynTyeFJZ3bO4arx2Ec3ps%3DBRMye4oGqrnfXDMf6OS67a3CYbVdJJ15Bq1GfZOYfaGufLjW44"

# Authenticate with Twitter API v2
client = tweepy.Client(bearer_token=bearer_token)

# 1️⃣ User Input for Topic
topic = input("Enter the topic you want to analyze (e.g., COVID vaccine, AI, elections): ")
query = f"{topic} -is:retweet lang:en"

# 2️⃣ Collect Tweets (max 100 per request on free tier)
tweets = client.search_recent_tweets(query=query,
                                     max_results=100,
                                     tweet_fields=["created_at", "text"])

tweet_list = [[tweet.text, tweet.created_at] for tweet in tweets.data]
df = pd.DataFrame(tweet_list, columns=['tweet', 'time'])

print("\n✅ Collected Tweets:", df.shape[0])
print(df.head())

# 3️⃣ Preprocessing
def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+", "", text)      # remove links
    text = re.sub(r"@\w+|#\w+", "", text)           # remove mentions & hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)         # remove punctuation/numbers
    text = text.lower().strip()
    return text

df['clean_tweet'] = df['tweet'].apply(clean_tweet)

# 4️⃣ Sentiment Analysis
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

# 5️⃣ Results Summary
print("\n📊 Sentiment Counts:")
print(df['Sentiment'].value_counts())

# Pie Chart
df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%',
                                        colors=['green', 'red', 'gray'],
                                        figsize=(6,6))
plt.title(f"Public Sentiment on '{topic}'")
plt.ylabel("")
plt.show()

# 6️⃣ Word Clouds
positive_text = " ".join(df[df['Sentiment']=="Positive"]['clean_tweet'])
negative_text = " ".join(df[df['Sentiment']=="Negative"]['clean_tweet'])

wc_pos = WordCloud(width=600, height=400, background_color="white", colormap="Greens").generate(positive_text)
wc_neg = WordCloud(width=600, height=400, background_color="white", colormap="Reds").generate(negative_text)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(wc_pos)
plt.axis("off")
plt.title("Positive Tweets")

plt.subplot(1,2,2)
plt.imshow(wc_neg)
plt.axis("off")
plt.title("Negative Tweets")
plt.show()

