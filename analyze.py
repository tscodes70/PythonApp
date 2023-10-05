import csv, pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
import sys, os

# Prepare VADER downloadables
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

INPUTFILE = "data.csv"
ENCODING = "utf8"

OUTPUTFILE = 'outputdata.csv'

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
SUMMARYLENGTH = 4  # Adjust the length of the summary as needed
OUTPUTHEADER = ['Hotel Name', 'Province', 'Postal','Category','SubCategory', 'Compound Sentiment', 'Review Summary', 'Total Reviews', 'Popular Keywords','Average Rating']

def getKeywords(hotelReviewList:list) -> list:
    """
    Retrieves and returns the top keywords of a list of reviews
    for a single Hotel

    Parameters:
    hotelReviewList (list): A list of reviews for a single hotel

    Returns:
    A sorted list of keywords based on frequency in descending order
    """
    testremovestop = []
    # Remove Stop words
    for item in hotelReviewList:
        # Tokenize the text
        words = nltk.word_tokenize(item)

        # Get a list of English stop words
        stop_words = set(stopwords.words('english'))

        # Remove stop words from the text
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Join the filtered words back into a sentence
        filtered_text = ' '.join(filtered_words)
        testremovestop.append(filtered_text)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    df = pd.DataFrame({'Review': testremovestop})
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Review'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    return sorted(list(zip(tfidf_df.columns.tolist(),tfidf_df.sum().tolist())),key=lambda x:x[1],reverse=True)[:5]



def analyzeIndividualReviews(processedData:pd.DataFrame,sia:SentimentIntensityAnalyzer):
    """
    Analyzes individual reviews using VADER Sentiment Analysis then
    1. Summarizes top 4 hotel reviews.
    2. Retrieves top 5 Keywords of hotel reviews

    This function performs a analysis of individual hotel reviews by summarizing top 4 hotel reviews
    and calling the 'getKeywords' function to extract top 5 keywords of reviews. Finally, both
    of these data are added into the intial Dataframe

    Parameters:
    processedData (pd.DataFrame): A Pandas Dataframe of individual hotel reviews to be analyzed.
    sia (SentimentIntensityAnalyzer): A initialized SentimentIntensityAnalyzer

    Returns:
    None
    """
    reviewSummaryOfHotel,keywordOfHotelReview = [],[]

    for reviewsGroupedByHotel in processedData['Review Summary']:
        sentimentScores = []

        reviewListEachHotel = reviewsGroupedByHotel.split('<SPLIT>')
        for review in reviewListEachHotel:
            sentimentScores.append(sia.polarity_scores(review))
        
        # Retrieving top keywords associated with top reviews
        keywordOfHotelReview.append(getKeywords(reviewListEachHotel))

        # Retrieving top 4 reviews for each hotel
        rankedReviews = sorted(enumerate(sentimentScores), key=lambda x: x[1]['compound'], reverse=True)
       
        top_sentences = nlargest(SUMMARYLENGTH, rankedReviews, key=lambda x: x[1]['compound'])
        reviewSummaryOfHotel.append([reviewListEachHotel[index] for index, _ in sorted(top_sentences)])

    processedData["Popular Keywords"],processedData['Review Summary'] = keywordOfHotelReview,reviewSummaryOfHotel

def analyzeConcatReviews(processedData:pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes a list of text reviews using VADER Sentiment Analysis.

    This function takes in a list of text reviews and analyzes them
    using the VADER Sentiment Analysis tool, which is implemented
    using the Natural Language Toolkit (NLTK). 
    
    It calculates the compound sentiment scores the reviews of each 
    hotel, formats them and returns a Pandas Dataframe containing 
    ['Hotel Name', 'Province', 'Postal','Category','SubCategory', 'Compound Sentiment', 
    'Review Summary', 'Total Reviews', 'Popular Keywords','Average Rating']

    Parameters:
    reviews (list of str): A list of text reviews to be analyzed.

    Returns:
    processedData (pd.Dataframe): A Pandas Dataframe after analyzing
    """

    sia = SentimentIntensityAnalyzer()
    # Tokenizing and analyzing reviews (for each hotel, concatted reviews)
    processedData['Sentiment Scores'] = processedData['Review'].apply(lambda x: sia.polarity_scores(x))
    # Retrieve Compound Sentiment
    processedData['Compound Sentiment'] = processedData['Sentiment Scores'].apply(lambda x: x['compound'])
    # Tokenizing and analyzing each review (for Review Summarization)
    analyzeIndividualReviews(processedData,sia)
    # Formatting for output
    processedData['Average Rating'] = processedData['Average Rating'].round(2)
    processedData['Total Reviews'] = processedData['Review'].apply(lambda x: len(x.split('<SPLIT> ')))
    
    return processedData


def processDataFromCsv(cleanCsvFilename:str) -> pd.DataFrame:
    """
    Reads data from a CSV file, preprocesses it, and groups it based on 
    'Hotel Name', 'Province', 'Postal', 'Category', 'SubCategory'.

    This function performs data preprocessing and grouping tasks,
    preparing it for the NLTK Analyzer to analyze all of the reviews for
    a specific hotel.

    Parameters:
    cleanCsvFilename (str): The filename of the CSV file containing the cleaned data to be processed.

    Returns:
    pandas.DataFrame: A DataFrame containing the preprocessed and grouped data based on specified criteria.
    """
    HotelName,HotelProvince,HotelPostal,HotelPrimaryCategory,HotelSubCategory,HotelReviews,ReviewTitle,HotelRating = [], [], [], [], [], [], [],[]
    GroupFilter = ['Hotel Name', 'Province', 'Postal', 'Category', 'SubCategory']
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

    with open(cleanCsvFilename, encoding=ENCODING) as f:
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

    return pd.DataFrame(data).groupby(GroupFilter).agg({'Review': lambda x: '<SPLIT> '.join(x), 'Review Summary': lambda y: '<SPLIT> '.join(y),'Average Rating': 'mean'}).reset_index()
    
def main():
    data = processDataFromCsv(INPUTFILE)
    analyzeConcatReviews(data)[OUTPUTHEADER].to_csv(OUTPUTFILE)
    
try:       
    main()
    print(f"======= Analyzing Completed =======")

except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)