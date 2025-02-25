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

# Step 1: User input
symbol = input("Enter the stock or crypto symbol (e.g., AAPL, BTC): ").upper()

# Step 2: Calculate date range (past 7 days)
to_date = datetime.now()
from_date = to_date - timedelta(days=7)
from_timestamp = int(from_date.timestamp())  # CryptoCompare uses Unix timestamps

# Step 3: Fetch news from CryptoCompare API (no API key needed for public endpoint)
url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={symbol}&lTs={from_timestamp}"
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error fetching news: {e}")
    exit()

# Check if articles exist
if data.get('Data') is None or not data['Data']:
    print(f"No articles found for {symbol} in the past 7 days.")
    exit()

articles = data["Data"]

# Step 4: Analyze sentiment and collect data
data_list = []
for article in articles:
    headline = article["title"] or "No title available"
    date_unix = article["published_on"]
    date = datetime.fromtimestamp(date_unix)
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

"Over the past 7 days, Bitcoin (BTC) headlines averaged 26.00% bullish, 24.00% bearish, and 50.00% neutral, reflecting a largely neutral market sentiment with balanced optimism and caution. On February 18, the only day with data, bullish sentiment reached 26.00% (13 headlines), driven by positive developments like XRP ETF progress, MicroStrategy’s steady 478,740 BTC holdings, and El Salvador’s increased Bitcoin purchases. Bearish sentiment hit 24.00% (12 headlines), fueled by uncertainties around FTX payouts, Bitcoin’s $110K roadblocks, and Solana’s crash. With no multi-day trends due to single-day data, the market appears in a holding pattern, awaiting clearer catalysts."