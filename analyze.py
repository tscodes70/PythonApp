import nltk
import csv, pandas as pd
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
HOTELRATING = "reviews.rating"

RATINGMAX = 5

def nltkAnalyzer(processedData):
    """
    This functions takes in a List of text reviews
    and analyzes them using the VADER Sentiment Analysis
    which is using the Natural Language ToolKit(NLTK)
    and prints the compound sentiment scores of the reviews
    70% Accuracy
    """
     # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Tokenize and analyze the reviews
    processedData['Sentiment Scores'] = processedData['Review'].apply(lambda x: sia.polarity_scores(x))

    # Extract sentiment scores
    processedData['Compound Sentiment'] = processedData['Sentiment Scores'].apply(lambda x: x['compound'])

    outputCsv(processedData[['Hotel Name', 'Province', 'Postal','Category','SubCategory', 'Compound Sentiment', 'Total Reviews', 'Average Rating']])
    
    # Cannot output reviews in csv as excel has char limit of 32k
    # outputCsv(processedData[['Hotel Name', 'Province', 'Postal', 'Review', 'Compound Sentiment', 'Total Reviews']])

def processDataFromCsv(csvFilename):
    HotelName,HotelProvince,HotelPostal,HotelPrimaryCategory,HotelSubCategory,HotelReviews,HotelRating = [], [], [], [], [], [],[]

    with open(csvFilename, encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            HotelName.append(row[HOTELNAME])
            HotelProvince.append(row[HOTELPROVINCE])
            HotelPostal.append(row[HOTELPOSTALCODE])
            HotelPrimaryCategory.append(row[HOTELPCATEGORY])
            HotelSubCategory.append(row[HOTELSCATEGORY])
            HotelReviews.append(row[HOTELREVIEWS])
            HotelRating.append(float(row[HOTELRATING]))

    data = {
    'Hotel Name': HotelName,
    'Province': HotelProvince,
    'Postal':HotelPostal,
    'Category':HotelPrimaryCategory,
    'SubCategory':HotelSubCategory,
    'Review': HotelReviews,
    'Average Rating': HotelRating
    }
    df = pd.DataFrame(data)
    """
        Used chatgpt to generate {'Review': lambda x: '<SPLIT> '.join(x), 'Average Rating': 'mean'}
        Reason: Encountering alot of errors & unsure of the formatting when applying 2 functions 
        while only resetting index once to dataframe.

        Original statement pre-chatgpt commented.

    """
    groupedData = df.groupby(['Hotel Name', 'Province', 'Postal', 'Category', 'SubCategory']).agg({'Review': lambda x: '<SPLIT> '.join(x), 'Average Rating': 'mean'}).reset_index()
    #groupedData = df.groupby(['Hotel Name', 'Province','Postal','Category','SubCategory'])['Review'].agg(lambda x: '<SPLIT> '.join(x)).reset_index()
    
    # Formatting for output csv
    groupedData['Average Rating'] = groupedData['Average Rating'].round(2)
    groupedData['Total Reviews'] = groupedData['Review'].apply(lambda x: len(x.split('<SPLIT> ')))

    nltkAnalyzer(groupedData)

def outputCsv(data):
    data.to_csv(OUTPUTFILE)
    
def main():
    processDataFromCsv(FILENAME)
    
main()

