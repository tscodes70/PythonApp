import nltk
import csv, pandas as pd
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from heapq import nlargest

# Prepare VADER pre-trained model
nltk.download('punkt')
nltk.download('vader_lexicon')

FILENAME = "data.csv"
ENCODING = "utf8"

OUTPUTFILE = 'outputdata.csv'
# PSENTIMENT = 0.05
# NSENTIMENT = -0.05

#CSV Headers
HOTELNAME =  "name"
HOTELPROVINCE = "province"
HOTELPOSTALCODE = "postalCode"
HOTELPCATEGORY = "categories"
HOTELSCATEGORY = "primaryCategories"
HOTELREVIEWS = "reviews.text"
REVIEWTITLE = "reviews.title"
HOTELRATING = "reviews.rating"

RATINGMAX = 5

def nltkAnalyzer(processedData) -> None:
    """
    This functions takes in a List of text reviews
    and analyzes them using the VADER Sentiment Analysis
    which is using the Natural Language ToolKit(NLTK)
    and prints the compound sentiment scores of the reviews
    70% Accuracy
    """
    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    # Tokenizing and analyzing reviews (for each hotel, concatted reviews)
    processedData['Sentiment Scores'] = processedData['Review'].apply(lambda x: sia.polarity_scores(x))
    # Retrieve Comp Senti
    processedData['Compound Sentiment'] = processedData['Sentiment Scores'].apply(lambda x: x['compound'])

    # Tokenizing and analyzing each review (for Review Summarization)
    processedData = getIndividualReviewSA(processedData)
        
    # Formatting for output csv
    processedData['Average Rating'] = processedData['Average Rating'].round(2)
    processedData['Total Reviews'] = processedData['Review'].apply(lambda x: len(x.split('<SPLIT> ')))
    
    outputCsv(processedData[['Hotel Name', 'Province', 'Postal','Category','SubCategory', 'Compound Sentiment', 'Review Summary', 'Total Reviews', 'Average Rating']])
    
    # Cannot output reviews in csv as excel has char limit of 32k
    # outputCsv(processedData[['Hotel Name', 'Province', 'Postal', 'Review', 'Compound Sentiment', 'Total Reviews']])

def getIndividualReviewSA(processedData):
    """
    This function analyzes each individual review using the
    VADER Sentiment Analysis
    which is using the Natural Language ToolKit(NLTK)
    retrieves the top 4 reviews by sorting the compounded sentiment,
    then generates a summary using these 4 reviews.
    """
    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    summary = []
    for reviewsGroupedByHotel in processedData['Review Summary']:
        sentimentScores = []

        reviewList = reviewsGroupedByHotel.split('<SPLIT>')
        for review in reviewList:
            sentimentScores.append(sia.polarity_scores(review))
   
        rankedReviews = sorted(enumerate(sentimentScores), key=lambda x: x[1]['compound'], reverse=True)
        summary_length = 4  # Adjust the length of the summary as needed
        top_sentences = nlargest(summary_length, rankedReviews, key=lambda x: x[1]['compound'])
        summary.append([reviewList[index] for index, _ in sorted(top_sentences)])

    processedData['Review Summary'] = summary
    return processedData
    


def processDataFromCsv(csvFilename:str):
    """
    This function reads data from a csv,
    then puts it into a pandas dataframe
    and returns the dataframe
    """
    HotelName,HotelProvince,HotelPostal,HotelPrimaryCategory,HotelSubCategory,HotelReviews,ReviewTitle,HotelRating = [], [], [], [], [], [], [],[]

    with open(csvFilename, encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            HotelName.append(row[HOTELNAME])
            HotelProvince.append(row[HOTELPROVINCE])
            HotelPostal.append(row[HOTELPOSTALCODE])
            HotelPrimaryCategory.append(row[HOTELPCATEGORY])
            HotelSubCategory.append(row[HOTELSCATEGORY])
            HotelReviews.append(row[HOTELREVIEWS])
            ReviewTitle.append(row[REVIEWTITLE])
            HotelRating.append(float(row[HOTELRATING]))

    data = {
    'Hotel Name': HotelName,
    'Province': HotelProvince,
    'Postal':HotelPostal,
    'Category':HotelPrimaryCategory,
    'SubCategory':HotelSubCategory,
    'Review': HotelReviews,
    'Review Summary': ReviewTitle,
    'Average Rating': HotelRating
    }
    df = pd.DataFrame(data)
    """
        Used chatgpt to generate {'Review': lambda x: '<SPLIT> '.join(x), 'Average Rating': 'mean'}
        Reason: Encountering alot of errors & unsure of the formatting when applying 2 functions 
        while only resetting index once to dataframe.

        Original statement pre-chatgpt commented.

    """

    groupedData = df.groupby(['Hotel Name', 'Province', 'Postal', 'Category', 'SubCategory']).agg({'Review': lambda x: '<SPLIT> '.join(x), 'Review Summary': lambda y: '<SPLIT> '.join(y),'Average Rating': 'mean'}).reset_index()
    #groupedData = df.groupby(['Hotel Name', 'Province','Postal','Category','SubCategory'])['Review'].agg(lambda x: '<SPLIT> '.join(x)).reset_index()
    return groupedData

def outputCsv(data):
    data.to_csv(OUTPUTFILE)
    
def main():
    data = processDataFromCsv(FILENAME)
    nltkAnalyzer(data)
    
main()

