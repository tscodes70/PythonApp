# -*- coding: utf-8 -*-
"""
Created on Sun Oct 8 00:51:41 2023

@author: Timothy
"""
import spacy,csv,ast
import pandas as pd
import globalVar, traceback
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

amenities_list = [
    'adults only',
    'air conditioning',
    'bar / lounge',
    'bathrobes',
    'beach',
    'bicycle tours',
    'bicycles available',
    'black-out curtains',
    'board games / puzzles',
    'body wrap',
    'breakfast buffet',
    'bridal suite',
    'cable / satellite tv',
    'car hire',
    'children activities (kid / family friendly)',
    'city view',
    'coffee / tea maker',
    'coffee / tea making facilities',
    'complimentary toiletries',
    'conference facilities',
    'convenience store',
    'desk',
    'dining area',
    'dog / pet friendly',
    'electric vehicle charging station',
    'extra long beds',
    'family rooms',
    'fireplace',
    'fitness centre with gym / workout room',
    'flatscreen tv',
    'free breakfast',
    'free high speed internet (wifi)',
    'free parking',
    'free public parking nearby',
    'free shuttle or taxi services',
    'gym',
    'highchairs available',
    'housekeeping',
    'kids stay free',
    'landmark view',
    'laundry service',
    'minibar',
    'night club / dj',
    'non-smoking hotel',
    'non-smoking rooms',
    'ocean view',
    'pets allowed',
    'pool',
    'private check-in / check-out',
    'refrigerator',
    'safe',
    'seating area',
    'shared lounge / tv area',
    'spa',
    'taxi service',
    'walk-in shower',
    'wifi',
    'fitness centre'
]


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
    dateAdded, dateUpdated, address, categories, primaryCategories, city, country, keys, latitude, longitude, name, postalCode, province, reviews_date, reviews_dateSeen, reviews_rating, reviews_sourceURLs, reviews_text, reviews_title, reviews_userCity, reviews_userProvince, reviews_username, sourceURLs, websites, reviews_cleantext, reviews_summary, average_rating, amenities, prices,star_rating = ([] for _ in range(30))
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
    'reviews.summary': reviews_summary,
    'average.rating': average_rating,
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

            amenities.append((row[globalVar.AMENITIES]))
            prices.append(row[globalVar.PRICES])

            reviews_cleantext.append(row[globalVar.REVIEWS_CLEANTEXT])
            reviews_summary.append(row[globalVar.REVIEWS_TEXT])
            average_rating.append(float(row[globalVar.REVIEWS_RATING]))
            # star_rating.append(row['star.rating'])
    return pd.DataFrame(data)

def groupDataframe(processedData:pd.DataFrame,filter:list) -> pd.DataFrame:
    return processedData.groupby(filter)

def predictiveModelling(dataframe:pd.DataFrame):
    for index, row in dataframe.iterrows():
        review = row[globalVar.REVIEWS_TEXT]
        amenities = row[globalVar.AMENITIES]
        if(len(review)<=1000000):
            doc = nlp(str(review))
            if len(amenities) == 2:
                # Extract mentions of amenities
                extracted_amenities  = list(set(token.text.lower().capitalize() for token in doc if token.text.lower() in amenities_list))  # Customize amenities list
                dataframe.at[index, globalVar.AMENITIES] = f"[{', '.join(extracted_amenities)}]"

        print(f"Processed {index + 1} hotels out of {len(dataframe[globalVar.REVIEWS_TEXT])}")

    return dataframe

# def process_amenities_and_predict(dataframe):
#     # Create binary columns for each facility in the dataset
#     facilities = list(set(facility for amenities_list in dataframe[globalVar.AMENITIES] for facility in amenities_list))
#     for facility in facilities:
#         dataframe[facility] = dataframe[globalVar.AMENITIES].apply(lambda x: 1 if facility in x else 0)

#     # Select binary facility features and target variable (prices)
#     X = dataframe[facilities]
#     y = dataframe[globalVar.PRICES]

#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Create and train a Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Make predictions on the testing set
#     y_pred = model.predict(X_test)
#     print(y_pred)
#     # Evaluate the model's performance
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)

#     # print("Mean Absolute Error:", mae)
#     # print("Mean Squared Error:", mse)

#     return dataframe

def handleMissingData():
    df = processDataFromCsv(globalVar.MDINPUTFULLFILE)
    # adf = groupDataframe(df,[globalVar.NAME, globalVar.PROVINCE, globalVar.POSTALCODE, globalVar.CATEGORIES, globalVar.PRIMARYCATEGORIES, globalVar.AMENITIES, globalVar.PRICES]).agg({
    # globalVar.REVIEWS_TEXT: lambda x: x,
    # globalVar.REVIEWS_CLEANTEXT: lambda z:z,
    # }).reset_index()

    pdf = groupDataframe(df,[globalVar.NAME, globalVar.PROVINCE, globalVar.POSTALCODE, globalVar.CATEGORIES, globalVar.PRIMARYCATEGORIES, globalVar.AMENITIES, globalVar.PRICES]).agg({
    globalVar.REVIEWS_TEXT: lambda x: ' '.join(x),
    globalVar.REVIEWS_CLEANTEXT: lambda z: ' '.join(z),
    }).reset_index()

    # adf[globalVar.PRICES] = adf[globalVar.PRICES].fillna(0)
    # process_amenities_and_predict(adf)

    fdf = predictiveModelling(pdf)
    newfdf = fdf[[globalVar.POSTALCODE,globalVar.AMENITIES]]
    df.update(newfdf)
    df[globalVar.PRICES] = df[globalVar.PRICES].replace('', 0)
    df.to_csv(globalVar.MDOUTPUTFULLFILE)
