import nltk
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Prepare VADER trained models
nltk.download('vader_lexicon')
# nltk.download('movie_reviews')

FILENAME = "data.csv"
ENCODING = "utf8"
PSENTIMENT = 0.05
NSENTIMENT = -0.05

def totalRows(filename):
    with open(filename, encoding=ENCODING) as f:
        print(sum(1 for line in f))

def loadData(filename: str) -> list:
    dataList = []
    with open(filename, encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataList.append(row['reviews.text'])      
    return dataList


def nltkAnalyzer(reviewstxt:list[str]):
    """
    This functions takes in a List of text reviews
    and analyzes them using the VADER Sentiment Analysis
    which is using the Natural Language ToolKit(NLTK)
    and prints the sentiment scores of the reviews

    """
    vaderAnalyzer = SentimentIntensityAnalyzer()

    # Analyze each review
    for review in reviewstxt:
        sentiment_scores = vaderAnalyzer.polarity_scores(review)
        print(f"Review: {review}")
        print(f"Sentiment Scores: {sentiment_scores}")

        # Determine sentiment by calculating compound score
        compound_score = sentiment_scores['compound']
        if compound_score >= PSENTIMENT:
            sentiment = "Positive"
        elif compound_score <= NSENTIMENT:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        print(f"Sentiment: {sentiment}\n")

def main():
    reviewstxt = loadData(FILENAME)
    nltkAnalyzer(reviewstxt)
    
main()