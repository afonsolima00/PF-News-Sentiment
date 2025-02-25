import requests
import csv
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Ensure VADER lexicon is available
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    print("VADER lexicon not found. Downloading...")
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

# Step 1: Set up API key and user input
api_key = "844aaabd70544f04ba59b34b2ac4ceb1"  # Replace with your free NewsAPI key from newsapi.org
symbol = input("Enter the stock or crypto symbol (e.g., AAPL, BTC): ")

# Step 2: Calculate date range (past 7 days)
to_date = datetime.now()
from_date = to_date - timedelta(days=7)
from_date_str = from_date.strftime("%Y-%m-%d")
to_date_str = to_date.strftime("%Y-%m-%d")

# Define financial news sources
domains = "bloomberg.com,reuters.com,cnbc.com,wsj.com,marketwatch.com"

# Construct NewsAPI URL
url = f"https://newsapi.org/v2/everything?q={symbol}&from={from_date_str}&to={to_date_str}&domains={domains}&apiKey={api_key}"

# Step 3: Fetch news headlines
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error fetching news: {e}")
    exit()

# Check if articles exist
if 'articles' not in data or not data['articles']:
    print(f"No articles found for {symbol} in the past 7 days.")
    exit()

articles = data["articles"]

# Step 4: Analyze sentiment and collect data
data_list = []
for article in articles:
    headline = article["title"] or "No title available"
    date_str = article["publishedAt"]
    try:
        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        date = to_date  # Fallback to current date if parsing fails
    scores = sia.polarity_scores(headline)
    compound = scores['compound']
    sentiment = 'bullish' if compound > 0.05 else 'bearish' if compound < -0.05 else 'neutral'
    data_list.append({'date': date, 'headline': headline, 'sentiment': sentiment})

# Create a DataFrame
df = pd.DataFrame(data_list)

# Step 5: Store results in a CSV
csv_filename = f"{symbol}_sentiment.csv"
df.to_csv(csv_filename, index=False)
print(f"Sentiment data saved to {csv_filename}")

# Step 6: Analyze sentiment patterns
df['date_only'] = df['date'].dt.date
daily_sentiment = df.groupby('date_only')['sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100

# Calculate average and standard deviation
average_sentiment = daily_sentiment.mean()
std_sentiment = daily_sentiment.std()

# Identify spikes
bullish_spikes = daily_sentiment[daily_sentiment.get('bullish', pd.Series()) > average_sentiment.get('bullish', 0) + std_sentiment.get('bullish', 0)]
bearish_spikes = daily_sentiment[daily_sentiment.get('bearish', pd.Series()) > average_sentiment.get('bearish', 0) + std_sentiment.get('bearish', 0)]

# Count headlines per day
headline_counts = df.groupby('date_only').size()

# Step 7: Generate summary
print("\nSentiment Analysis Summary (Past 7 Days):")
print(f"Average sentiment: {average_sentiment.get('bullish', 0):.2f}% bullish, "
      f"{average_sentiment.get('bearish', 0):.2f}% bearish, "
      f"{average_sentiment.get('neutral', 0):.2f}% neutral")

if not bullish_spikes.empty or not bearish_spikes.empty:
    print("\nNotable Sentiment Spikes:")
    for date in bullish_spikes.index:
        count = headline_counts.get(date, 0)
        print(f"- {date}: Bullish sentiment spiked to {bullish_spikes.loc[date, 'bullish']:.2f}% ({count} headlines)")
    for date in bearish_spikes.index:
        count = headline_counts.get(date, 0)
        print(f"- {date}: Bearish sentiment spiked to {bearish_spikes.loc[date, 'bearish']:.2f}% ({count} headlines)")
else:
    print("\nNo significant sentiment spikes detected.")

'''
Over the past 7 days, NVIDIA-related news headlines showed a balanced sentiment mix, averaging 26.11% bullish, 24.17% bearish, and 49.72% neutral. Two notable spikes emerged: on February 18, bullish sentiment surged to 66.67% (3 headlines), likely driven by positive developments like Musk’s xAI Grok 3 release and Cramer’s optimistic stock picks. Conversely, on February 21, bearish sentiment spiked to 100% (1 headline), tied to concerns over Apple’s Vision Pro app shortage, indirectly reflecting tech sector pressures. The remaining days leaned neutral, with no strong trends, suggesting a cautious market mood ahead of NVIDIA’s earnings.
'''