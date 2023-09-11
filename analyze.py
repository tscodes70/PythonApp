import nltk
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Prepare VADER pre-trained model
nltk.download('vader_lexicon')

FILENAME = "data.csv"
ENCODING = "utf8"
PSENTIMENT = 0.05
NSENTIMENT = -0.05

# def totalRows(filename):
#     with open(filename, encoding=ENCODING) as f:
#         print(sum(1 for line in f))

def getUniqueHotelTuple(hotelNameList,hotelPcodeList):
    """
    Make NameList & PostalList unique and turns it into a tuple
    while preserving the order.
    """
    # Create empty lists to maintain the order for unique names and codes
    unique_hotel_names = []
    unique_hotel_codes = []

    # Iterate through both lists, adding unique pairs to the new lists
    for name, code in zip(hotelNameList, hotelPcodeList):
        if (name, code) not in zip(unique_hotel_names, unique_hotel_codes):
            unique_hotel_names.append(name)
            unique_hotel_codes.append(code)
    
    return tuple(zip(unique_hotel_names, unique_hotel_codes))


def getSortedData(filename):
    """
    This functions takes in data from a csv and sorts
    by grouping each hotel's reviews to its own postal code.
    """
    hotelPcodeList = []
    hotelNameList =[]
    reviewList = []

    with open(filename, encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            hotelPcodeList.append(row['postalCode'])
            hotelNameList.append(row['name'])
            reviewList.append(row['reviews.text'])
        
    hotelUniqueTuple = getUniqueHotelTuple(hotelNameList,hotelPcodeList)
    
    return [[hotel[0], hotel[1],[review for i, review in enumerate(reviewList) if hotelPcodeList[i] == hotel[1]]] for hotel in hotelUniqueTuple]


def nltkAnalyzer(sortedData):
    """
    This functions takes in a List of text reviews
    and analyzes them using the VADER Sentiment Analysis
    which is using the Natural Language ToolKit(NLTK)
    and prints the sentiment scores of the reviews
    70% Accuracy
    """
    vaderAnalyzer = SentimentIntensityAnalyzer()
    posSentCounter = 0
    negSentCounter = 0
    neuSentCounter = 0

    # Analyze each review
    for hotelData in sortedData:
        sentiment_scores = vaderAnalyzer.polarity_scores(hotelData)
        #print(f"Review: {review}")
        #print(f"Sentiment Scores: {sentiment_scores}")

        # Determine sentiment by calculating compound score
        compound_score = sentiment_scores['compound']
        if compound_score >= PSENTIMENT:
            sentiment = "Positive"
            posSentCounter+=1

        elif compound_score <= NSENTIMENT:
            sentiment = "Negative"
            negSentCounter+=1
        else:
            sentiment = "Neutral"
            neuSentCounter+=1

        #print(f"Sentiment: {sentiment}\n")
    
    print(f"Positive: {posSentCounter} Negative: {negSentCounter} Neutral: {neuSentCounter}")
    print("====================================================\n\n\n")

def printData(sortedData):
    for hotelData in sortedData:
        print("====================================================")
        print(f"Hotel: {hotelData[0]}")
        print(f"Hotel Code: {hotelData[1]}")
        nltkAnalyzer(hotelData[2])
            
def main():
    printData(getSortedData(FILENAME))

    
main()

