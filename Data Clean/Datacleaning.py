import pandas as pd
import schedule
import time

timestamp = time.strftime("%Y%m%d-%H%M%S")
def dataclean():
   df = pd.read_csv('outputdata.csv')
   # Get first n rows
   df.head()
   # Print information about the csv
   df.info()
   # Sort Hotel Name in ascending order
   df = df.sort_values(by=['Hotel Name'], ascending = True)
   # Remove missing values
   df = df.dropna()
   # Remove duplicate rows from csv
   df.drop_duplicates(inplace = True)
   # Replace Hotels to Hotel
   df['Category'] = df['Category'].str.replace('Hotels', 'Hotel', case=False)
   # Replace Casinos to Casino
   df['Category'] = df['Category'].str.replace('Casinos', 'Casino', case=False)
   # Replace Motels to Motel
   df['Category'] = df['Category'].str.replace('Motels', 'Motel', case=False)
   # Replace Hotel and Motel to Hotel & Motel
   df['Category'] = df['Category'].str.replace('Hotel and Motel', 'Hotel & Motel', case=False)
   # Replace Hotel Motel to Hotel & Motel
   df['Category'] = df['Category'].str.replace('Hotel Motel', 'Hotel & Motel', case=False)
   # Remove duplicate categories
   df['Category'] = df['Category'].str.split(',').apply(lambda x: ', '.join(set(x)))
   # Add space
   df['Category'] = df['Category'].str.replace(',', ', ')
   print(df)
   # Save to Clean.csv
   df.to_csv('Clean.csv', index=False, float_format='%.0f')

# Scheduled task
schedule.every(0.1).minutes.do(dataclean)
while True:
   schedule.run_pending()
   time.sleep(1)
