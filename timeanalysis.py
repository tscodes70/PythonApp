import pandas as pd
import holidays
from scipy.stats import pearsonr
from datetime import datetime
import calendar

# Load the dataset into a DataFrame
df = pd.read_csv("analyzedreviews_10-Oct.csv")

# Split the timestamp at 'T' and keep only the date portion
df['reviews.date'] = df['reviews.date'].str.split('T', expand=True)[0]

# Convert the 'dateAdded' column to datetime format
df['reviews.date'] = pd.to_datetime(df['reviews.date'], format='%Y-%m-%d')

#output_file = "data.xlsx"
#df.to_excel(output_file, index=False)

# Function to check if a date is a holiday for the corresponding country and year
def holiday_name(row):
    year = row['reviews.date'].year
    country = row['country'].strip()

    # Calculate holidays for the specific year and country
    country_holidays = getattr(holidays, country)(years=year)

    if row['reviews.date'].date() in country_holidays:
        return country_holidays.get(row['reviews.date'].date())
    else:
        return 'Not Holiday'  # Return None if it's not a holiday

# Create a new column to indicate the holiday name or None
df['is_holiday'] = df.apply(holiday_name, axis=1)

# Convert 'is_holiday' to binary values (1 for holiday, 0 for not holiday)
df['is_holiday'] = df['is_holiday'].apply(lambda x: 1 if x != 'Not Holiday' else 0)

# Group by name and calculate correlation for each name
name_correlations = df.groupby('name').apply(lambda group: group['Compound Sentiment'].corr(group['is_holiday']))

# Replace NaN values with a custom message
name_correlations.fillna('Not enough data', inplace=True)

# Modify 'is_holiday' to 0 for 'Not Holiday' instances
# df['is_holiday'] = df['is_holiday'].apply(lambda x: 1 if x == 'Not Holiday' else 0)

# Group by name and calculate correlation for each name for 'is_holiday' equal to 0
# name_correlations_not_holiday = df.groupby('name').apply(lambda group: group['Compound Sentiment'].corr(group['is_holiday']))

# Save the correlation results to an Excel file
# correlation_excel_path = "name_correlations.xlsx"
# name_correlations_df = name_correlations.reset_index()  # Resetting index to have 'name' as a column
# name_correlations_df.columns = ['name', 'Correlation']  # Renaming columns

# with pd.ExcelWriter(correlation_excel_path) as writer:
#     name_correlations_df.to_excel(writer, sheet_name='Name_Correlations')

#print(name_correlations)

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

# Create a new column to indicate the season
df['season'] = df['reviews.date'].apply(get_season)

# Save the DataFrame with the new columns to an Excel file
# output_file = "data_with_holidays_seasons.xlsx"
# df.to_excel(output_file, index=False)

# print(f'Data with holidays saved to {output_file}')


# Assuming you have a column 'Season' indicating the season for each review
# Modify this condition based on your dataset structure
df['is_winter'] = df['season'].apply(lambda x: 1 if x == 'Winter' else 0)
df['is_spring'] = df['season'].apply(lambda x: 1 if x == 'Spring' else 0)
df['is_summer'] = df['season'].apply(lambda x: 1 if x == 'Summer' else 0)
df['is_autumn'] = df['season'].apply(lambda x: 1 if x == 'Autumn' else 0)


# Group by hotel and calculate correlation for each hotel
hotel_correlations_winter = df.groupby('name').apply(lambda group: group['Compound Sentiment'].corr(group['is_winter']))
hotel_correlations_spring = df.groupby('name').apply(lambda group: group['Compound Sentiment'].corr(group['is_spring']))
hotel_correlations_summer = df.groupby('name').apply(lambda group: group['Compound Sentiment'].corr(group['is_summer']))
hotel_correlations_autumn = df.groupby('name').apply(lambda group: group['Compound Sentiment'].corr(group['is_autumn']))
# Add more correlations for other seasons as needed

# Replace NaN values with a custom message
hotel_correlations_winter.fillna('Data not enough', inplace=True)
hotel_correlations_spring.fillna('Data not enough', inplace=True)
hotel_correlations_summer.fillna('Data not enough', inplace=True)
hotel_correlations_autumn.fillna('Data not enough', inplace=True)
# Add more replacements for other seasons as needed

#print("Correlation with Winter:")
#print(hotel_correlations_winter)

#print("\nCorrelation with Spring:")
#print(hotel_correlations_spring)
# Add more prints for other seasons as neede

# Save the DataFrame with the new columns to an Excel file
# output_file = "data_with_holidays_seasons.xlsx"
# df.to_excel(output_file, index=False)

# print(f'Data with holidays saved to {output_file}')