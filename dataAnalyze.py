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
from heapq import nlargest,nsmallest
import globalVar,traceback
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time,ast
from scipy.stats import pearsonr
import numpy as np


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
    return (tokenizer.decode(outputs[0], skip_special_tokens=True)).capitalize()

def analyzeIndividualReviews(processedData:pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes individual reviews using VADER Sentiment Analysis then
    1. Summarizes top 2 hotel reviews.
    2. Summarizes worst 2 hotel reviews.
    3. Retrieves top 10 Keywords of hotel reviews

    This function performs a analysis of individual hotel reviews by 
    calling the 'getSummary' function to summarize the top 2 hotel reviews and 
    calling the 'getKeywords' function to extract top 10 keywords of reviews. 
    Finally, both of these data are added into the intial Dataframe

    Parameters:
    processedData (pd.DataFrame): A Pandas Dataframe of individual hotel reviews to be analyzed.

    Returns:
    processedData (pd.DataFrame): A Pandas Dataframe appended with data of Review Summary & Popular Keywords
    """
    goodSummaryOfHotel, badSummaryOfHotel, keywordsOfHotel = [],[],[]
    sia = SentimentIntensityAnalyzer()
    for index, reviewsGroupedByHotel in enumerate(processedData[globalVar.REVIEWS_TEXT]):
        reviewListEachHotel = reviewsGroupedByHotel.split('<SPLIT>')
        # Sorting Reviews from best to worst
        rankedReviews = sorted(reviewListEachHotel, key=lambda x: sia.polarity_scores(x)['compound'], reverse=True)
        # Retrieve top 2 reviews sorted by sentiment score
        best_sentences = nlargest(globalVar.REVIEWSUMMAX, rankedReviews, key=lambda x: sia.polarity_scores(x)['compound'])
        # Retrieve worst 2 reviews sorted by sentiment score
        worst_sentences = nsmallest(globalVar.REVIEWSUMMAX, rankedReviews, key=lambda x: sia.polarity_scores(x)['compound'])
        # Summarize the sorted reviews using transformer
        goodSummaryOfHotel.append(getSummary(" ".join(best_sentences)))
        badSummaryOfHotel.append(getSummary(" ".join(worst_sentences)))

        print(f"Summary of {index + 1} hotels out of {len(processedData[globalVar.REVIEWS_TEXT])} generated")

    
    keywordsOfHotel = [getKeywords(item) for item in processedData[globalVar.REVIEWS_CLEANTEXT]]

    processedData[globalVar.POPULAR_KEYWORDS],processedData[globalVar.GREVIEWS_SUMMARY],processedData[globalVar.BREVIEWS_SUMMARY] = keywordsOfHotel,goodSummaryOfHotel,badSummaryOfHotel
    return processedData

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
    dateAdded, dateUpdated, address, categories, primaryCategories, city, country, keys, latitude, longitude, name, postalCode, province, reviews_date, reviews_dateSeen, reviews_rating, reviews_sourceURLs, reviews_text, reviews_title, reviews_userCity, reviews_userProvince, reviews_username, sourceURLs, websites, reviews_cleantext, reviews_summary, average_rating, amenities, prices = ([] for _ in range(29))
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

    'amenities' : amenities,
    'prices': prices,

    'reviews.cleantext': reviews_cleantext,
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

            amenities.append(ast.literal_eval(row[globalVar.AMENITIES]))
            prices.append(float(row[globalVar.PRICES]))

            reviews_cleantext.append(row[globalVar.REVIEWS_CLEANTEXT])
            reviews_summary.append(row[globalVar.REVIEWS_TEXT])
            average_rating.append(float(row[globalVar.REVIEWS_RATING]))
    return pd.DataFrame(data)

def initiateAnalysis(data:pd.DataFrame, OUTPUTREVIEWFULLFILE,OUTPUTHOTELFULLFILE):
    # Prepare Base Analysis
    sia = SentimentIntensityAnalyzer()
    data[globalVar.SENTIMENT_SCORE] = data[globalVar.REVIEWS_TEXT].apply(lambda x: sia.polarity_scores(x))
    data[globalVar.COMPOUND_SENTIMENT_SCORE] = data[globalVar.SENTIMENT_SCORE].apply(lambda x: x['compound'])
    gProcessedData = data.copy()
    # Export individual review analysis
    gProcessedData[[globalVar.NAME, globalVar.PROVINCE, globalVar.COUNTRY, globalVar.REVIEWS_DATE, globalVar.REVIEWS_TEXT, globalVar.REVIEWS_RATING, globalVar.COMPOUND_SENTIMENT_SCORE]].to_csv(OUTPUTREVIEWFULLFILE)
    
    # Export average compound analysis grouped by hotel
    hProcessedData = gProcessedData.copy()
    hProcessedData[globalVar.AMENITIES] = hProcessedData[globalVar.AMENITIES].astype(str)
    hProcessedData = groupDataframe(hProcessedData,[globalVar.NAME, globalVar.PROVINCE, globalVar.POSTALCODE, globalVar.CATEGORIES, globalVar.PRIMARYCATEGORIES, globalVar.AMENITIES]).agg({
    globalVar.REVIEWS_TEXT: lambda x: '<SPLIT> '.join(x),
    globalVar.REVIEWS_CLEANTEXT: lambda z: ' '.join(z),
    globalVar.AVERAGE_RATING: 'mean',
    globalVar.PRICES: 'mean',
    globalVar.COMPOUND_SENTIMENT_SCORE: 'mean'
    }).reset_index()
    hProcessedData = analyzeIndividualReviews(hProcessedData)
    hProcessedData[globalVar.AVERAGE_RATING] = hProcessedData[globalVar.AVERAGE_RATING].round(2)
    hProcessedData[globalVar.REVIEWS_TOTAL] = hProcessedData[globalVar.REVIEWS_TEXT].apply(lambda x: int(len(x.split('<SPLIT> '))))
    hProcessedData[globalVar.REVIEWS_LENGTH] = hProcessedData[globalVar.REVIEWS_TEXT].apply(lambda x: int(len(' '.join(x.split('<SPLIT> ')))))
    hProcessedData[globalVar.REVIEWS_TOTAL] = pd.to_numeric(hProcessedData[globalVar.REVIEWS_TOTAL], errors='coerce')
    hProcessedData = hProcessedData[hProcessedData[globalVar.REVIEWS_TOTAL] > 5]

    hProcessedData[globalVar.ANALYSISOUTPUTHEADER].to_csv(OUTPUTHOTELFULLFILE)
    return gProcessedData,hProcessedData

def averageRatingCorrelation(processedData:pd.DataFrame):
    df = processedData[[globalVar.AVERAGE_RATING,globalVar.COMPOUND_SENTIMENT_SCORE]].groupby(globalVar.AVERAGE_RATING)[globalVar.COMPOUND_SENTIMENT_SCORE].mean().reset_index()
    correlation,pvalue = pearsonr(processedData[globalVar.AVERAGE_RATING], processedData[globalVar.COMPOUND_SENTIMENT_SCORE])
    print(f"==== Correlation of Average Rating ==== \n{correlation}")
    return correlation

def averageReviewLengthCorrelation(processedData:pd.DataFrame):
    filterprocessedData = processedData.copy()
    filterprocessedData[globalVar.AVERAGE_REVIEWS_LENGTH] = filterprocessedData[globalVar.REVIEWS_LENGTH] / filterprocessedData[globalVar.REVIEWS_TOTAL]
    filterprocessedData = filterprocessedData[~filterprocessedData[globalVar.AVERAGE_REVIEWS_LENGTH].isin([np.inf, -np.inf])]
    filterprocessedData = filterprocessedData.dropna(subset=[globalVar.AVERAGE_REVIEWS_LENGTH])
    correlation,pvalue = pearsonr(filterprocessedData[globalVar.AVERAGE_REVIEWS_LENGTH], filterprocessedData[globalVar.COMPOUND_SENTIMENT_SCORE])
    print(f"==== Correlation of Average Review Length ==== \n{correlation}")
    return correlation

def amenitiesCorrelation(processedData:pd.DataFrame):
    filterprocessedData = processedData.copy()
    uAmenties = list(set(amenity for amenity_list in filterprocessedData[globalVar.AMENITIES] for amenity in amenity_list))
    # Initialize a dictionary to store the binary columns for each amenity
    amenity_columns = {}
    # Create binary columns for each unique amenity
    for amenity in uAmenties:
        filterprocessedData[amenity] = filterprocessedData[globalVar.AMENITIES].apply(lambda x: 1 if amenity in x else 0)
        amenity_columns[amenity] = filterprocessedData[amenity]
    correlation = filterprocessedData[uAmenties].corrwith(filterprocessedData[globalVar.COMPOUND_SENTIMENT_SCORE])
    print(f'==== Correlation of Amenities ==== \n{correlation}')
    return correlation

def priceCorrelation(processedData:pd.DataFrame):
    filterprocessedData = processedData.copy()
    filterprocessedData = filterprocessedData.dropna(subset=[globalVar.PRICES])
    filterprocessedData[globalVar.PRICES] = filterprocessedData[globalVar.PRICES].replace('', 0)
    correlation,pvalue = pearsonr(pd.to_numeric(filterprocessedData[globalVar.PRICES]), filterprocessedData[globalVar.COMPOUND_SENTIMENT_SCORE])
    print(f'==== Correlation of Price ==== \n{correlation}')
    return correlation

def provinceCorrelation(processedData:pd.DataFrame):
    # Step 1: One-Hot Encoding
    province_dummies = pd.get_dummies(processedData[globalVar.PROVINCE])

    # Step 2: Calculate Correlation
    correlation = province_dummies.apply(lambda x: x.corr(processedData[globalVar.COMPOUND_SENTIMENT_SCORE]))
    print(f'==== Correlation of Province ==== \n{correlation}')
    return correlation

def getWeights(totalCorrelations:dict):
    # Step 1: Normalize Correlation Coefficients
    absolute_coefficients = {var: abs(coeff) for var, coeff in totalCorrelations.items()}
    total_absolute = sum(absolute_coefficients.values())
    normalized_weights = {var: coeff / total_absolute for var, coeff in absolute_coefficients.items()}

    # Step 2: Use Normalized Coefficients as Weights
    weights = normalized_weights

    # Print the normalized weights
    return weights

def predictSentiment(weight):
    # data_point = {
    # 'average.reviews.length': 120,
    # 'average.rating': 1,
    # 'prices': 400,
    # 'NY': 1,
    # 'CA': 0,
    # 'LA': 0,
    # 'HI': 0,
    # 'GA': 0,
    # 'WA': 0,
    # 'PA': 0,
    # 'TX': 0,
    # 'Wifi': 0,
    # 'Board games / puzzles': 1,
    # 'Air conditioning': 0,
    # 'Non-smoking rooms': 0,
    # 'Family rooms': 1,
    # 'Seating area': 0,
    # 'Free parking': 0,
    # 'Non-smoking hotel': 0,
    # 'Private check-in / check-out': 0,
    # 'Suites': 0,
    # 'Free High Speed Internet (WiFi)': 0,
    # 'Shared lounge / TV area': 0,
    # 'AZ': 0,
    # 'MO': 0,
    # 'MD': 0,
    # 'FL': 0,
    # 'NJ': 0,
    # 'IL': 0,
    # 'MA': 0,
    # 'NV': 0
    # }
    data_point = {
        'average.reviews.length': 0, #above average
        'average.rating': 1, #below average
        'prices': 1, #below average
        'amenities': 0, #below average
        'province': 0
    }
    # Calculate the weighted sum for the data point
    weighted_sum = sum(weight[var] * data_point[var] for var in weight)
    print(weighted_sum)
    # Interpret the result (e.g., set a threshold)
    threshold = 0.5  # You can adjust the threshold based on your problem
    predicted_sentiment = 'Positive' if weighted_sum > threshold else 'Negative'

    # Print the predicted sentiment
    print("Predicted Sentiment:", predicted_sentiment)

def dataAnalysis(INPUTFULLFILE,OUTPUTREVIEWFULLFILE,OUTPUTHOTELFULLFILE,GETCORRELATIONS):
    gProcessedData,hProcessedData = initiateAnalysis(processDataFromCsv(INPUTFULLFILE),OUTPUTREVIEWFULLFILE,OUTPUTHOTELFULLFILE)

    if GETCORRELATIONS:
        totalCorrelations = {}

        # Correlation analysis
        provinceCorr = provinceCorrelation(gProcessedData)
        amenitiesCorr = amenitiesCorrelation(gProcessedData)
        totalCorrelations[globalVar.AVERAGE_RATING] = averageRatingCorrelation(hProcessedData)
        totalCorrelations[globalVar.AVERAGE_REVIEWS_LENGTH] = averageReviewLengthCorrelation(hProcessedData)
        totalCorrelations[globalVar.PRICES] = priceCorrelation(gProcessedData)
        totalCorrelations[globalVar.PROVINCE] = provinceCorr.mean()
        totalCorrelations[globalVar.AMENITIES] = amenitiesCorr.mean()
        totalCorrelations.update(provinceCorr.to_dict())
        totalCorrelations.update(amenitiesCorr.to_dict())
        sortedTotalCorrelations= dict(sorted(totalCorrelations.items(), key=lambda item: item[1], reverse=True))
        sortedTotalCorrelationsDf = pd.DataFrame(list(sortedTotalCorrelations.items()), columns=[globalVar.CORRVARIABLE, globalVar.CORRCOEFFICIENT])
        sortedTotalCorrelationsDf.to_csv(globalVar.CORRFULLFILE, index=False)
        #Series
        # provinceCorrelation(gProcessedData)
        # amenitiesCorrelation(gProcessedData)

        # weight = getWeights(sortedTotalCorrelations)
        # predictSentiment(weight)
