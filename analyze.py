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
import numpy as np
import globalVar,traceback


# Prepare downloadbles
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

def getKeywords(hotelReviewList:list) -> list:
    """
    Retrieves and returns the top keywords of a list of reviews
    for a single Hotel

    Parameters:
    hotelReviewList (list): A list of reviews for a single hotel

    Returns:
    A sorted list of keywords based on frequency in descending order
    """
    hotelSplitList = hotelReviewList.split(' ')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    df = pd.DataFrame({globalVar.REVIEWS_TEXT: hotelSplitList})
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[globalVar.REVIEWS_TEXT])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    return sorted(list(zip(tfidf_df.columns.tolist(),tfidf_df.sum().tolist())),key=lambda x:x[1],reverse=True)[:globalVar.KEYWORDMAX]

def getSummary(reviews):
    # Tokenize the text into sentences
    doc = nlp(reviews)
    sentences = [sent.text for sent in doc.sents]

    # Calculate the importance score of each sentence based on length
    sentence_scores = {sentence: len(sentence) for sentence in sentences}

    # Select the top N sentences with the highest importance scores
    top_sentences = nlargest(globalVar.REVIEWSUMMAX, sentence_scores, key=sentence_scores.get)

    # Create the summary by joining selected sentences
    summary = ' '.join(top_sentences)

    return summary

def analyzeIndividualReviews(processedData:pd.DataFrame) -> pd.DataFrame:
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
    for reviewsGroupedByHotel in processedData[globalVar.REVIEWS_SUMMARY]:
        reviewListEachHotel = reviewsGroupedByHotel.split('<SPLIT>')
        # Sorting Reviews from best to worst
        rankedReviews = sorted(reviewListEachHotel, key=lambda x: sia.polarity_scores(x)['compound'], reverse=True)
        # Retrieve top 4 reviews sorted by sentiment score
        top_sentences = nlargest(globalVar.REVIEWSUMMAX, rankedReviews, key=lambda x: sia.polarity_scores(x)['compound'])

        # Summarize the sorted reviews using spaCy
        summary = getSummary(" ".join(top_sentences))
        reviewSummaryOfHotel.append(summary)
    
    for item in processedData[globalVar.REVIEWS_CLEANTEXT]:
        keywordOfHotelReview.append(getKeywords(item)) 

    processedData[globalVar.POPULAR_KEYWORDS],processedData[globalVar.REVIEWS_SUMMARY] = keywordOfHotelReview,reviewSummaryOfHotel
    return processedData

def analyzeCorrelations(processedData:pd.DataFrame, variable):
    df = processedData[[variable,globalVar.COMPOUND_SENTIMENT_SCORE]].groupby(variable)[globalVar.COMPOUND_SENTIMENT_SCORE].mean().reset_index()
    correlation = df[variable].corr(df[globalVar.COMPOUND_SENTIMENT_SCORE])
    print(correlation)

def groupDataframe(processedData:pd.DataFrame,filter:list) -> pd.DataFrame:
    return processedData.groupby(filter)

def processDataFromCsv(cleanCsvFilename:str) -> pd.DataFrame:
    """
    Reads data from a CSV file, preprocesses it, and groups it based on 
    globalVar.NAME, globalVar.PROVINCE, globalVar.POSTALCODE, globalVar.CATEGORIES, globalVar.PRIMARYCATEGORIES.

    This function performs data preprocessing and grouping tasks,
    preparing it for the NLTK Analyzer to analyze all of the reviews for
    a specific hotel.

    Parameters:
    cleanCsvFilename (str): The filename of the CSV file containing the cleaned data to be processed.

    Returns:
    pandas.DataFrame: A DataFrame containing the preprocessed and grouped data based on specified criteria.
    """
    dateAdded, dateUpdated, address, categories, primaryCategories, city, country, keys, latitude, longitude, name, postalCode, province, reviews_date, reviews_dateSeen, reviews_rating, reviews_sourceURLs, reviews_text, reviews_title, reviews_userCity, reviews_userProvince, reviews_username, sourceURLs, websites, reviews_cleantext, reviews_summary, average_rating = ([] for _ in range(27))
    data = {
    'dateAdded': dateAdded,
    'dateUpdated': dateUpdated,
    'address': address,
    'categories': categories,
    'primaryCategories': primaryCategories,
    'city': city,
    'country': country,
    'keys': keys,
    'latitude': latitude,
    'longitude': longitude,
    'name': name,
    'postalCode': postalCode,
    'province': province,
    'reviews.date': reviews_date,
    'reviews.dateSeen': reviews_dateSeen,
    'reviews.rating': reviews_rating,
    'reviews.sourceURLs': reviews_sourceURLs,
    'reviews.text': reviews_text,
    'reviews.title': reviews_title,
    'reviews.userCity': reviews_userCity,
    'reviews.userProvince': reviews_userProvince,
    'reviews.username': reviews_username,
    'sourceURLs': sourceURLs,
    'websites': websites,

    'reviews.cleantext': reviews_cleantext,
    'reviews.summary': reviews_summary,
    'average.rating': average_rating
    }


    with open(cleanCsvFilename, encoding=globalVar.ANALYSISENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dateAdded.append(row[globalVar.DATEADDED])
            dateUpdated.append(row[globalVar.DATEUPDATED])
            address.append(row[globalVar.ADDRESS])
            categories.append(row[globalVar.CATEGORIES])
            primaryCategories.append(row[globalVar.PRIMARYCATEGORIES])
            city.append(row[globalVar.CITY])
            country.append(row[globalVar.COUNTRY])
            keys.append(row[globalVar.KEYS])
            latitude.append(row[globalVar.LATITUDE])
            longitude.append(row[globalVar.LONGITUDE])
            name.append(row[globalVar.NAME])
            postalCode.append(row[globalVar.POSTALCODE])
            province.append(row[globalVar.PROVINCE])
            reviews_date.append(row[globalVar.REVIEWS_DATE])
            reviews_dateSeen.append(row[globalVar.REVIEWS_DATESEEN])
            reviews_rating.append(float(row[globalVar.REVIEWS_RATING]))
            reviews_sourceURLs.append(row[globalVar.REVIEWS_SOURCEURLS])
            reviews_text.append(row[globalVar.REVIEWS_TEXT])
            reviews_title.append(row[globalVar.REVIEWS_TITLE])
            reviews_userCity.append(row[globalVar.REVIEWS_USERCITY])
            reviews_userProvince.append(row[globalVar.REVIEWS_USERPROVINCE])
            reviews_username.append(row[globalVar.REVIEWS_USERNAME])
            sourceURLs.append(row[globalVar.SOURCEURLS])
            websites.append(row[globalVar.WEBSITES])

            reviews_cleantext.append(row[globalVar.REVIEWS_CLEANTEXT])
            reviews_summary.append(row[globalVar.REVIEWS_TEXT])
            average_rating.append(float(row[globalVar.REVIEWS_RATING]))
    return pd.DataFrame(data)

def initiateAnalysis(data:pd.DataFrame):
    # Prepare Base Analysis
    sia = SentimentIntensityAnalyzer()
    data[globalVar.SENTIMENT_SCORE] = data[globalVar.REVIEWS_TEXT].apply(lambda x: sia.polarity_scores(x))
    data[globalVar.COMPOUND_SENTIMENT_SCORE] = data[globalVar.SENTIMENT_SCORE].apply(lambda x: x['compound'])
    gProcessedData = data.copy()
    # Export individual review analysis
    gProcessedData[[globalVar.NAME, globalVar.PROVINCE, globalVar.COUNTRY, globalVar.REVIEWS_DATE, globalVar.REVIEWS_TEXT, globalVar.COMPOUND_SENTIMENT_SCORE]].to_csv(globalVar.ANALYSISOUTPUTBYREVIEWS)
    
    # Export average compound analysis grouped by hotel
    hProcessedData = groupDataframe(gProcessedData.copy(),[globalVar.NAME, globalVar.PROVINCE, globalVar.POSTALCODE, globalVar.CATEGORIES, globalVar.PRIMARYCATEGORIES]).agg({
    globalVar.REVIEWS_TEXT: lambda x: '<SPLIT> '.join(x),
    globalVar.REVIEWS_SUMMARY: lambda y: '<SPLIT> '.join(y),
    globalVar.REVIEWS_CLEANTEXT: lambda z: ' '.join(z),
    globalVar.AVERAGE_RATING: 'mean',
    globalVar.COMPOUND_SENTIMENT_SCORE: 'mean'
    }).reset_index()
    hProcessedData = analyzeIndividualReviews(hProcessedData)
    hProcessedData[globalVar.AVERAGE_RATING] = hProcessedData[globalVar.AVERAGE_RATING].round(2)
    hProcessedData[globalVar.REVIEWS_TOTAL] = hProcessedData[globalVar.REVIEWS_TEXT].apply(lambda x: len(x.split('<SPLIT> ')))
    hProcessedData[globalVar.ANALYSISOUTPUTHEADER].to_csv(globalVar.ANALYSISOUTPUTBYHOTEL)

    return gProcessedData

def main():
    gProcessedData = initiateAnalysis(processDataFromCsv(globalVar.ANALYSISINPUTFILE))
    
    # Correlation analysis
    analyzeCorrelations(gProcessedData,globalVar.AVERAGE_RATING)
    # analyzeCorrelations(gProcessedData,globalVar.PROVINCE)
    # analyzeCorrelations(gProcessedData,globalVar.CATEGORIES) (budget or luxury)
    # analyzeCorrelations(gProcessedData,'Amenities') (facilities)
    # gProcessedData["ReviewDateFormatted"] = pd.to_datetime(gProcessedData[globalVar.REVIEWS_DATE])
    # analyzeCorrelations(gProcessedData,'ReviewDateFormatted')
    # analyzeCorrelations(gProcessedData,'ReviewLength')
    # analyzeCorrelations(gProcessedData,'Keywords')
    # analyzeCorrelations(gProcessedData,'Price')
    #.to_csv("outputdata2.csv")
    
    
try:       
    main()
    print(f"======= Analyzing Completed =======")

except:
    traceback.print_exc() 