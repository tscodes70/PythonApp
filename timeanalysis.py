import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import holidays
from datetime import date


# Load the dataset into a DataFrame
df = pd.read_csv("data.csv")

# Split the timestamp at 'T' and keep only the date portion
df['reviews.date'] = df['reviews.date'].str.split('T', expand=True)[0]

# Convert the 'dateAdded' column to datetime format
df['reviews.date'] = pd.to_datetime(df['reviews.date'], format='%Y-%m-%d')
#print(df['reviews.date'])

# Function to check if a date is a holiday for the corresponding country and year
def holiday_name(row):
    year = row['reviews.date'].year
    country = row['country']

    # Calculate holidays for the specific year and country
    country_holidays = getattr(holidays, country)(years=year)

    if row['reviews.date'].date() in country_holidays:
        return country_holidays.get(row['reviews.date'].date())
    else:
        return 'None'  # Return None if it's not a holiday

# Create a new column to indicate the holiday name or None
df['is_holiday'] = df.apply(holiday_name, axis=1)


# Save the DataFrame with the new columns to an Excel file
output_file = "data_with_holidays.xlsx"
df.to_excel(output_file, index=False)

print(f'Data with holidays saved to {output_file}')
