import pandas as pd
import schedule
import time
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
nltk.download('punkt')
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Sample text
text = "This is an example sentence with some stop words."

# Tokenize the text
words = nltk.word_tokenize(text)

# Get a list of English stop words
stop_words = set(stopwords.words('english'))

# Remove stop words and create a filtered list of words
filtered_words = [word for word in words if word.lower() not in stop_words]

# Join the filtered words back into a sentence
filtered_text = ' '.join(filtered_words)

print(filtered_text)

# Define a regex pattern to match emoji symbols
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U0001F004-\U0001F0CF"  # CJK Compatibility Ideographs Supplement
                           u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
                           u"\U0001F910-\U0001F93E"  # Emoticons (faces)
                           u"\U0001F940-\U0001F970"  # Emoticons (clothing)
                           u"\U0001F980-\U0001F991"  # Emoticons (animals)
                           u"\U0001F9A0"  # Emoticon (brain)
                           u"\U0001F9C0"  # Emoticon (face with medical mask)
                           u"\U0001F9E0"  # Emoticon (mechanical arm)
                           u"\U0001F9F0"  # Emoticon (tooth)
                           u"\U0001FA70-\U0001FA74"  # Zodiacal symbols
                           u"\U0001FA78-\U0001FA7A"  # Astrological signs
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251"  # Enclosed characters
                           u"\U00002000-\U0000206F"  # General Punctuation
                           u"\U00002070-\U0000209F"  # Superscripts and Subscripts
                           u"\U000020D0-\U000020FF"  # Combining Diacritical Marks for Symbols
                           u"\U00002100-\U0000214F"  # Letterlike Symbols
                           u"\U00002160-\U0000218F"  # Number Forms
                           "]+", flags=re.UNICODE)

def dataclean():
   # Read the CSV file with 'utf-8-sig' encoding
   df = pd.read_csv('test_4oct.csv', encoding='utf-8-sig')
   # Get first n rows
   df.head()
   # Print information about the csv
   df.info()
   # Sort Hotel Name in ascending order
   df = df.sort_values(by=['Ratings'], ascending = True)
   # Remove missing values
   df = df.dropna()
   # Remove duplicate rows from csv
   df.drop_duplicates(inplace = True)
   df = df.rename(columns={'Unnamed: 0': 'ID'})
   # Add new columns
   df['ID'] = range(1, len(df) + 1)
   df['Username'] = 'NA'
   df['UserCity'] = 'NA'
   df['UserProvince'] = 'NA'
   df['UserCountry'] = 'NA'
   column_order = ['ID', 'Date', 'Hotel Name', 'Address', 'Province', 'Postal', 'Country', 'Ratings', 'Category', 'Review Tittle', 'Review', 'Username', 'UserCity', 'UserProvince', 'UserCountry', 'Website URL']
   # Rearrange the columns according to the specified order
   df = df[column_order]
   print(df)
   df['Ratings'] = df['Ratings'].apply(lambda x: x // 10)
   # Remove emoji symbols from the column
   df['Review Tittle'] = df['Review Tittle'].apply(lambda x: emoji_pattern.sub(r'', x))
   df['Review'] = df['Review'].apply(lambda x: emoji_pattern.sub(r'', x))
   # Save to Clean.csv
   df.to_csv('Clean1.csv', index=False, float_format='%.0f', encoding='utf-8-sig')

# Scheduled task
schedule.every(0.1).minutes.do(dataclean)
while True:
   schedule.run_pending()
   time.sleep(1)
