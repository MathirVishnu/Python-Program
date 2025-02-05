import nltk 
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    
    # Determine overall sentiment
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, sentiment_scores

if __name__ == "__main__":
    # Example text
    text = input("Enter a sentence for sentiment analysis: ")
    sentiment, scores = analyze_sentiment(text)

    
    # Print results
    print(f"Sentiment: {sentiment}")
    print(f"Scores: {scores}")