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
import globalVar,traceback
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

# Prepare downloadbles
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-small")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def getKeywords(hotelReviewList:list) -> list:
    """
    Retrieves and returns the top keywords of all the reviews
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

def getSummary(topConcatReview:str) -> str:
    """
    Generates a summary based on the concatenated string of the
    top 4 reviews of a single Hotel using the Transformer library

    Parameters:
    topConcatReview (str): A str of top reviews for a single Hotel

    Returns:
    A generated summary of the top reviews of a hotel
    """
    # encode the text into tensor of integers using the appropriate tokenizer
    inputs = tokenizer.encode("summarize: " + topConcatReview, return_tensors="pt", max_length=512, truncation=True)    # generate the summarization output
    outputs = model.generate(
        inputs, 
        max_length=150, 
        min_length=15, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def analyzeIndividualReviews(processedData:pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes individual reviews using VADER Sentiment Analysis then
    1. Summarizes top 4 hotel reviews.
    2. Retrieves top 10 Keywords of hotel reviews

    This function performs a analysis of individual hotel reviews by 
    calling the 'getSummary' function to summarize the top 4 hotel reviews and 
    calling the 'getKeywords' function to extract top 10 keywords of reviews. 
    Finally, both of these data are added into the intial Dataframe

    Parameters:
    processedData (pd.DataFrame): A Pandas Dataframe of individual hotel reviews to be analyzed.

    Returns:
    processedData (pd.DataFrame): A Pandas Dataframe appended with data of Review Summary & Popular Keywords
    """
    reviewSummaryOfHotel,keywordOfHotelReview = [],[]
    sia = SentimentIntensityAnalyzer()
    for index, reviewsGroupedByHotel in enumerate(processedData[globalVar.REVIEWS_SUMMARY]):
        reviewListEachHotel = reviewsGroupedByHotel.split('<SPLIT>')
        # Sorting Reviews from best to worst
        rankedReviews = sorted(reviewListEachHotel, key=lambda x: sia.polarity_scores(x)['compound'], reverse=True)
        # Retrieve top 4 reviews sorted by sentiment score
        top_sentences = nlargest(globalVar.REVIEWSUMMAX, rankedReviews, key=lambda x: sia.polarity_scores(x)['compound'])

        # Summarize the sorted reviews using transformer
        reviewSummaryOfHotel.append(getSummary(" ".join(top_sentences)))

        print(f"Processed {index + 1} hotels out of {len(processedData[globalVar.REVIEWS_SUMMARY])}")

    
    keywordOfHotelReview = [getKeywords(item) for item in processedData[globalVar.REVIEWS_CLEANTEXT]]

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
    Reads data from a CSV file, processes it and puts it into a Pandas Dataframe.

    This function performs data preprocessing to prepare it 
    for the NLTK Analyzer.

    Parameters:
    cleanCsvFilename (str): The filename of the CSV file containing the cleaned data to be processed.

    Returns:
    pandas.DataFrame: A DataFrame containing the processed data read from the csv.
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
    sTime = time.time() 
    main()
    eTime = time.time()
    runtime = eTime - sTime
    print(f"======= Analyze Information =======")
    print(f"Runtime: {runtime} seconds")
    print(f"===================================")

except:
    traceback.print_exc() 