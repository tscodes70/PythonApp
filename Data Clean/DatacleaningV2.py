import pandas as pd
import schedule
import time
import nltk
from nltk.corpus import words, stopwords
import re
from langdetect import detect, LangDetectException
from spellchecker import SpellChecker


nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
timestamp = time.strftime("%Y%m%d-%H%M%S")


# Define function to remove titles
def filter_titles(title):
    # Define criteria
    keywords_to_remove = ["so", "soso", "Stole", "robbed"]

    # Check if any of the keywords are present in the title
    for keyword in keywords_to_remove:
        if re.search(r'\b{}\b'.format(keyword), title, flags=re.IGNORECASE):
            return None  # Return None to exclude the title
    return title

# Initialize the SpellChecker
spell = SpellChecker()

def correct_typos(text):
    # Tokenize the text into words
    words = text.split()
    # Correct typos in each word
    corrected_words = [spell.correction(word) for word in words]
    # Reconstruct the text with corrected words
    corrected_text = ' '.join(corrected_words)
    return corrected_text

# Define a function to read custom stopwords from a text file
def read_custom_stopwords(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        custom_stopwords = [line.strip() for line in file]
    return custom_stopwords

# Path to your custom stopwords text file
custom_stopwords_file = 'stopword_list.txt'

# Read custom stopwords from the text file
custom_stopwords = read_custom_stopwords(custom_stopwords_file)

# Combine custom stopwords with NLTK stopwords
all_stopwords = set(stopwords.words('english')).union(custom_stopwords)

# Remove stopwords and perform other checks
def process_review(review):
    # Tokenize the review
    words = nltk.word_tokenize(review)
    # Remove stopwords
    words = [word for word in words if word.lower() not in all_stopwords]
    # Check if the review is too short
    if len(words) < 5:
        return None  # Return None to indicate that the review should be excluded
    else:
        return ' '.join(words)  # Join the remaining words back into a sentence

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
#Return true if its English, else false
def is_english(sentence):
    try:
        return detect(sentence) == 'en'
    except Exception as e:
        print(f"Error detecting language for '{sentence}': {str(e)}")
        return False
def dataclean():
   # Read the CSV file with 'utf-8-sig' encoding
   df = pd.read_csv('test_4oct.csv', encoding='utf-8-sig' and 'utf-8')
   # Get first n rows
   df.head()
   # Print information about the csv
   df.info()
   df = df.sort_values(by=['reviews.ratings'], ascending = True)
   # Remove missing values
   df = df.dropna()
   # Remove duplicate rows from csv
   df.drop_duplicates(inplace = True)
   df = df.rename(columns={'Unnamed: 0': 'id'})
   # Add new columns
   df['id'] = range(1, len(df) + 1)
   df['reviews.username'] = 'NA'
   df['reviews.userCity'] = 'NA'
   df['reviews.userProvince'] = 'NA'
   df['sourceURLs'] = 'NA'
   df['reviews.sourceURLs'] = 'NA'
   df['reviews.dateSeen'] = 'NA'
   df['dateUpdated'] = 'NA'
   df['primaryCategories'] = 'NA'
   df['keys'] = 'NA'
   df['latitude'] = 'NA'
   df['longitude'] = 'NA'
   df['reviews.date'] = 'NA'
   df['reviews.dateAdded'] = 'NA'
   df['reviews.cleantext'] = df['reviews.text']
   column_order = ['id', 'dateAdded', 'dateUpdated', 'address', 'categories', 'primaryCategories', 'city', 'country', 'keys', 'latitude', 'longitude', 'name', 'postalCode', 'province', 'reviews.date', 'reviews.dateSeen', 'reviews.ratings', 'reviews.sourceURLs', 'reviews.text', 'reviews.title', 'reviews.cleantext', 'reviews.userCity', 'reviews.userProvince', 'reviews.username', 'sourceURLs', 'websites', 'reviews.dateAdded']
   # Rearrange the columns according to the specified order
   df = df[column_order]
   df['reviews.ratings'] = df['reviews.ratings'].apply(lambda x: x // 10)
   # Remove emoji symbols from the column
   df['reviews.title'] = df['reviews.title'].apply(lambda x: emoji_pattern.sub(r'', x))
   df['reviews.text'] = df['reviews.text'].apply(lambda x: emoji_pattern.sub(r'', x))
   df['reviews.cleantext'] = df['reviews.cleantext'].apply(lambda x: emoji_pattern.sub(r'', x))
   #Keep rows with English sentences
   df = df[df['reviews.title'].apply(is_english)]
   # Apply the process_review function to the 'reviews.text' column
   df['reviews.cleantext'] = df['reviews.cleantext'].apply(process_review)
   # Remove rows where reviews.text is None (reviews that are too short or empty after removing stopwords)
   df = df.dropna(subset=['reviews.text'])
   df = df.dropna(subset=['reviews.cleantext'])
   # Filter out hotels with too few reviews
   few_reviews = df['name'].value_counts()
   df = df[df['name'].isin(few_reviews.index[few_reviews >= 5])]
   # Apply the filter_titles function to the reviews.title column
   df['reviews.title'] = df['reviews.title'].apply(filter_titles)
   df['reviews.cleantext'] = df['reviews.cleantext'].str.replace(r',+', '.,')
   df['reviews.cleantext'] = df['reviews.cleantext'].str.replace(r',+', '..')
   # Remove duplicate from reviews.cleantext
   df['reviews.cleantext'] = df['reviews.cleantext'].str.split(',').apply(lambda x: ', '.join(set(x)))
   # Drop rows with None (filtered titles)
   df = df.dropna(subset=['reviews.title'])
   df['id'] = range(1, len(df) + 1)

   print(df)
   # Save to Clean.csv
   df.to_csv('Clean2.csv', index=False, float_format='%.0f', encoding='utf-8-sig' and 'utf-8')

# Scheduled task
schedule.every(0.1).minutes.do(dataclean)
while True:
   schedule.run_pending()
   time.sleep(1)
