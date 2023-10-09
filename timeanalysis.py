import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import holidays
from datetime import date
import numpy as np

# Load the dataset into a DataFrame
df = pd.read_csv("output3.csv")

# Split the timestamp at 'T' and keep only the date portion
df['ReviewDate'] = df['ReviewDate'].str.split('T', expand=True)[0]

# Convert the 'dateAdded' column to datetime format
df['ReviewDate'] = pd.to_datetime(df['ReviewDate'], format='%Y-%m-%d')

# Function to check if a date is a holiday for the corresponding country and year
def holiday_name(row):
    year = row['ReviewDate'].year
    country = row['Country']

    # Calculate holidays for the specific year and country
    country_holidays = getattr(holidays, country)(years=year)

    if row['ReviewDate'].date() in country_holidays:
        return country_holidays.get(row['ReviewDate'].date())
    else:
        return 'Not Holiday'  # Return None if it's not a holiday

# Create a new column to indicate the holiday name or None
df['is_holiday'] = df.apply(holiday_name, axis=1)

#Function to determine what season the month is in
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
df['season'] = df['ReviewDate'].apply(get_season)

# Save the DataFrame with the new columns to an Excel file
output_file = "data_with_holidays_seasons.xlsx"
df.to_excel(output_file, index=False)

#print(f'Data with holidays saved to {output_file}')
