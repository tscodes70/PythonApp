import pandas as pd
import holidays
from datetime import date
import numpy as np
import globalVar



# Function to check if a date is a holiday for the corresponding country and year
def holiday_name(row):
    year = row[globalVar.REVIEWS_DATE].year
    country = row[globalVar.COUNTRY].strip()

    # Calculate holidays for the specific year and country
    country_holidays = getattr(holidays, country)(years=year)

    if row[globalVar.REVIEWS_DATE].date() in country_holidays:
        return country_holidays.get(row[globalVar.REVIEWS_DATE].date())
    else:
        return 'Not Holiday'  # Return None if it's not a holiday

# Function to determine what season the month is in
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def timeAnalysis():
    # Load the dataset into a DataFrame
    df = pd.read_csv(globalVar.ANALYSISREVIEWOUTPUTFULLFILE)

    # Split the timestamp at 'T' and keep only the date portion
    df[globalVar.REVIEWS_DATE] = df[globalVar.REVIEWS_DATE].str.split('T', expand=True)[0]

    # Convert the 'dateAdded' column to datetime format
    df[globalVar.REVIEWS_DATE] = pd.to_datetime(df[globalVar.REVIEWS_DATE], format='%Y-%m-%d')

    #output_file = "data.xlsx"
    #df.to_excel(output_file, index=False)
    # Create a new column to indicate the holiday name or None
    df[globalVar.IS_HOLIDAY] = df.apply(holiday_name, axis=1)

    # Convert globalVar.IS_HOLIDAY to binary values (1 for holiday, 0 for not holiday)
    df[globalVar.IS_HOLIDAY] = df[globalVar.IS_HOLIDAY].apply(lambda x: 1 if x != 'Not Holiday' else 0)

    # Group by name and calculate correlation for each name
    name_correlations = df.groupby(globalVar.NAME).apply(lambda group: group[globalVar.COMPOUND_SENTIMENT_SCORE].corr(group[globalVar.IS_HOLIDAY]))

    # Replace NaN values with a custom message
    # name_correlations.fillna('Not enough data', inplace=True)

    # Modify globalVar.IS_HOLIDAY to 0 for 'Not Holiday' instances
    # df[globalVar.IS_HOLIDAY] = df[globalVar.IS_HOLIDAY].apply(lambda x: 1 if x == 'Not Holiday' else 0)

    # Group by name and calculate correlation for each name for globalVar.IS_HOLIDAY equal to 0
    # name_correlations_not_holiday = df.groupby(globalVar.NAME).apply(lambda group: group[globalVar.COMPOUND_SENTIMENT_SCORE].corr(group[globalVar.IS_HOLIDAY]))

    # Save the correlation results to an Excel file
    # correlation_excel_path = "name_correlations.xlsx"
    # name_correlations_df = name_correlations.reset_index()  # Resetting index to have globalVar.NAME as a column
    # name_correlations_df.columns = [globalVar.NAME, 'Correlation']  # Renaming columns

    # with pd.ExcelWriter(correlation_excel_path) as writer:
    #     name_correlations_df.to_excel(writer, sheet_name='Name_Correlations')

    #print(name_correlations)

    # Create a new column to indicate the season
    df[globalVar.SEASON] = df[globalVar.REVIEWS_DATE].apply(get_season)

    # Save the DataFrame with the new columns to an Excel file
    # output_file = "data_with_holidays_seasons.xlsx"
    # df.to_excel(output_file, index=False)

    # print(f'Data with holidays saved to {output_file}')


    # Assuming you have a column globalVar.SEASON indicating the season for each review
    # Modify this condition based on your dataset structure
    df[globalVar.IS_WINTER] = df[globalVar.SEASON].apply(lambda x: 1 if x == 'Winter' else 0)
    df[globalVar.IS_SPRING] = df[globalVar.SEASON].apply(lambda x: 1 if x == 'Spring' else 0)
    df[globalVar.IS_SUMMER] = df[globalVar.SEASON].apply(lambda x: 1 if x == 'Summer' else 0)
    df[globalVar.IS_AUTUMN] = df[globalVar.SEASON].apply(lambda x: 1 if x == 'Autumn' else 0)


    # Group by hotel and calculate correlation for each hotel
    hotel_correlations_winter = df.groupby(globalVar.NAME).apply(lambda group: group[globalVar.COMPOUND_SENTIMENT_SCORE].corr(group[globalVar.IS_WINTER]))
    hotel_correlations_spring = df.groupby(globalVar.NAME).apply(lambda group: group[globalVar.COMPOUND_SENTIMENT_SCORE].corr(group[globalVar.IS_SPRING]))
    hotel_correlations_summer = df.groupby(globalVar.NAME).apply(lambda group: group[globalVar.COMPOUND_SENTIMENT_SCORE].corr(group[globalVar.IS_SUMMER]))
    hotel_correlations_autumn = df.groupby(globalVar.NAME).apply(lambda group: group[globalVar.COMPOUND_SENTIMENT_SCORE].corr(group[globalVar.IS_AUTUMN]))
    # Add more correlations for other seasons as needed

    # Replace NaN values with a custom message
    # hotel_correlations_winter.fillna('Data not enough', inplace=True)
    # hotel_correlations_spring.fillna('Data not enough', inplace=True)
    # hotel_correlations_summer.fillna('Data not enough', inplace=True)
    # hotel_correlations_autumn.fillna('Data not enough', inplace=True)
    # Add more replacements for other seasons as needed

    name_correlations = name_correlations.dropna()
    hotel_correlations_winter = hotel_correlations_winter.dropna()
    hotel_correlations_spring = hotel_correlations_spring.dropna()
    hotel_correlations_summer = hotel_correlations_summer.dropna()
    hotel_correlations_autumn = hotel_correlations_autumn.dropna()

    correlation_results = pd.DataFrame({
    'Hotel Name': hotel_correlations_winter.index,
    'Correlation (Winter)': hotel_correlations_winter.values,
    'Correlation (Spring)': hotel_correlations_spring.values,
    'Correlation (Summer)': hotel_correlations_summer.values,
    'Correlation (Autumn)': hotel_correlations_autumn.values,
    'Correlation': name_correlations
    })

    corr_dict = {}
    corr_dict['holiday'] = name_correlations.mean()
    corr_dict['winter'] = hotel_correlations_winter.mean()
    corr_dict['spring'] = hotel_correlations_spring.mean()
    corr_dict['summer'] = hotel_correlations_summer.mean()
    corr_dict['autumn'] = hotel_correlations_autumn.mean()

    corr_dictDf = pd.DataFrame(list(corr_dict.items()), columns=['Variables', 'Correlation Coefficient'])

    existing_data = pd.read_csv(globalVar.CORRFULLFILE)
    updated_data = existing_data.append(corr_dictDf, ignore_index=True)

    updated_data.to_csv(globalVar.CORRFULLFILE, index=False)

    #print("Correlation with Winter:")
    #print(hotel_correlations_winter)

    #print("\nCorrelation with Spring:")
    #print(hotel_correlations_spring)
    # Add more prints for other seasons as neede

    # Save the DataFrame with the new columns to an Excel file
    # output_file = "data_with_holidays_seasons.csv"
    # correlation_results.to_csv(output_file, index=False)

    # print(f'Data with holidays saved to {output_file}')
