# -*- coding: utf-8 -*-
"""
Created on Sun Sept 10 19:24:58 2023

@author: Timothy
"""
import csv, pandas as pd
import spacy
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

nlp = spacy.load('en_core_web_sm')

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
REVIEWDATE = "reviews.date"
HOTELRATING = "reviews.rating"

RATINGMAX = 5
SUMMARYLENGTH = 2  # Adjust the length of the summary as needed
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

def getSummary(reviews):
    # Tokenize the text into sentences
    doc = nlp(reviews)
    sentences = [sent.text for sent in doc.sents]

    # Calculate the importance score of each sentence based on length
    sentence_scores = {sentence: len(sentence) for sentence in sentences}

    # Select the top N sentences with the highest importance scores
    top_sentences = nlargest(SUMMARYLENGTH, sentence_scores, key=sentence_scores.get)

    # Create the summary by joining selected sentences
    summary = ' '.join(top_sentences)

    return summary

def analyzeIndividualReviews(processedData:pd.DataFrame):
    """
    Analyzes individual reviews using VADER Sentiment Analysis then
    1. Summarizes top 2 hotel reviews.
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
    sia = SentimentIntensityAnalyzer()


    for reviewsGroupedByHotel in processedData['Review Summary']:
        sentimentScores = []

        reviewListEachHotel = reviewsGroupedByHotel.split('<SPLIT>')
        for review in reviewListEachHotel:
            sentimentScores.append(sia.polarity_scores(review))

        # Sorting Reviews from best to worst
        rankedReviews = sorted(enumerate(sentimentScores), key=lambda x: x[1]['compound'], reverse=True)
        # Retrieve top 4 reviews sorted by sentiment score
        top_sentences = nlargest(SUMMARYLENGTH, rankedReviews, key=lambda x: x[1]['compound'])

        sortedReviewList = [reviewListEachHotel[index] for index, _ in sorted(top_sentences)]

        # Summarize the sorted reviews using spaCy
        summary = getSummary(" ".join(sortedReviewList))
        reviewSummaryOfHotel.append(summary)

    # Retrieving top keywords associated with top reviews (change to cleanedtext)
    for reviewsGroupedByHotel in processedData['Review Summary']:
        reviewListEachHotel = reviewsGroupedByHotel.split('<SPLIT>')
        keywordOfHotelReview.append(getKeywords(reviewListEachHotel)) 

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
    gProcessedData = processedData.agg({'Review': lambda x: '<SPLIT> '.join(x), 'Review Summary': lambda y: '<SPLIT> '.join(y),'Average Rating': 'mean'}).reset_index()
    
    sia = SentimentIntensityAnalyzer()
    # Tokenizing and analyzing reviews
    gProcessedData['Sentiment Scores'] = gProcessedData['Review'].apply(lambda x: sia.polarity_scores(x))
    # Retrieve Compound Sentiment
    gProcessedData['Compound Sentiment'] = gProcessedData['Sentiment Scores'].apply(lambda x: x['compound'])

    return gProcessedData


def analyzeCorrelations(processedData:pd.DataFrame, filter):
    return processedData[[filter,'Compound Sentiment']].groupby(filter)['Compound Sentiment'].mean().reset_index()


def groupDataframe(processedData:pd.DataFrame,filter:list) -> pd.DataFrame:
    return processedData.groupby(filter)

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
    HotelName,HotelProvince,HotelPostal,HotelPrimaryCategory,HotelSubCategory,HotelReviews,HotelReviewDate,ReviewTitle,HotelRating = [], [], [], [], [], [], [], [],[]
    data = {
    'Hotel Name': HotelName,
    'Province': HotelProvince,
    'Postal':HotelPostal,
    'Category':HotelPrimaryCategory,
    'SubCategory':HotelSubCategory,
    'Review': HotelReviews,
    'ReviewDate': HotelReviewDate,
    'Review Summary': HotelReviews,
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
            HotelReviewDate.append(row[REVIEWDATE])
            ReviewTitle.append(row[REVIEWTITLE])
            HotelRating.append(float(row[HOTELRATING]))

    return pd.DataFrame(data)
    
def main():
    data = processDataFromCsv(INPUTFILE)
    gProcessedData = analyzeConcatReviews(groupDataframe(data,['Hotel Name', 'Province', 'Postal', 'Category', 'SubCategory']))
    analyzeCorrelations(gProcessedData,'Province').to_csv("outputdata2.csv")

    # Tokenizing and analyzing each review (for Review Summarization)
    analyzeIndividualReviews(gProcessedData)

    # Formatting for output
    gProcessedData['Average Rating'] = gProcessedData['Average Rating'].round(2)
    gProcessedData['Total Reviews'] = gProcessedData['Review'].apply(lambda x: len(x.split('<SPLIT> ')))
    gProcessedData[OUTPUTHEADER].to_csv(OUTPUTFILE)
try:       
    main()
    print(f"======= Analyzing Completed =======")

except Exception as e:
    # exc_type, exc_obj, exc_tb = sys.exc_info()
    # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    # print(exc_type, fname, exc_tb.tb_lineno)
    print(e)