import csv,re, nltk
import pandas as pd
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
import globalVar


CUSTOMSTOPWORDS = r'C:\Users\anyho\Desktop\PythonProject\csvs\stopword_list.txt'

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

# Define a function to read custom stopwords from a text file
def read_custom_stopwords(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        custom_stopwords = [line.strip() for line in file]
    return custom_stopwords

# Define function to remove titles
def filter_titles(title):
    # Define criteria
    keywords_to_remove = ["so", "soso", "Stole", "robbed"]

    # Check if any of the keywords are present in the title
    for keyword in keywords_to_remove:
        if re.search(r'\b{}\b'.format(keyword), title, flags=re.IGNORECASE):
            return None  # Return None to exclude the title
    return title

def process_review(review):
    # Read custom stopwords from the text file
    custom_stopwords = read_custom_stopwords(CUSTOMSTOPWORDS)
    # Combine custom stopwords with default NLTK stopwords
    all_stopwords = set(stopwords.words('english')).union(custom_stopwords)
    # Tokenize review
    words = nltk.word_tokenize(review)
    # Remove stopwords
    words = [word for word in words if word.lower() not in all_stopwords]
    # Check if the review is too short
    if len(words) < 5:
        return None  # Return None to indicate that the review should be excluded
    else:
        return ' '.join(words)  # Join the remaining words back into a sentence

#Define function to ... after sentence
def remove_sentence(sentence):
    #Match regex pattern
    sentence_pattern = r'\.\.\.$'

    # Use re.sub to replace the sentence pattern with an empty string
    cleaned_sentence = re.sub(sentence_pattern, '', sentence)

    return cleaned_sentence

#Return true if its English, else false
def is_english(sentence):
    try:
        return detect(sentence) == 'en'
    except Exception as e:
        print(f"Error detecting language for '{sentence}': {str(e)}")
        return False
    
def categoryReplace(dataFrame:pd.DataFrame) -> pd.DataFrame:
    # Replace Hotels to Hotel
   dataFrame['categories'] = dataFrame['categories'].str.replace('Hotels', 'Hotel', case=False)
   # Replace Casinos to Casino
   dataFrame['categories'] = dataFrame['categories'].str.replace('Casinos', 'Casino', case=False)
   # Replace Motels to Motel
   dataFrame['categories'] = dataFrame['categories'].str.replace('Motels', 'Motel', case=False)
   # Replace Hotel and Motel to Hotel & Motel
   dataFrame['categories'] = dataFrame['categories'].str.replace('Hotel and Motel', 'Hotel & Motel', case=False)
   # Replace Hotel Motel to Hotel & Motel
   dataFrame['categories'] = dataFrame['categories'].str.replace('Hotel Motel', 'Hotel & Motel', case=False)
   # Remove duplicate categories
   dataFrame['categories'] = dataFrame['categories'].str.split(',').apply(lambda x: ', '.join(set(x)))
   # Add space
   dataFrame['categories'] = dataFrame['categories'].str.replace(',', ', ')
   return dataFrame

def removeStopwords(dataFrame:pd.DataFrame):
    # Apply the filter_titles function to the reviews.title column
   dataFrame['reviews.title'] = dataFrame['reviews.title'].apply(filter_titles)
   dataFrame['reviews.cleantext'] = dataFrame['reviews.cleantext'].str.replace(',', '')
   dataFrame['reviews.cleantext'] = dataFrame['reviews.cleantext'].str.replace('!', '')
   dataFrame['reviews.cleantext'] = dataFrame['reviews.cleantext'].str.replace(':', '')
   dataFrame['reviews.cleantext'] = dataFrame['reviews.cleantext'].str.replace('-', '')
   dataFrame['reviews.cleantext'] = dataFrame['reviews.cleantext'].str.replace(';', ',')
   dataFrame['reviews.cleantext'] = dataFrame['reviews.cleantext'].str.replace('.,', ',')
   dataFrame['reviews.cleantext'] = dataFrame['reviews.cleantext'].str.replace('(', '')
   dataFrame['reviews.cleantext'] = dataFrame['reviews.cleantext'].str.replace(')', '')
   return dataFrame



def dataCleaning(scrapeDataframe:pd.DataFrame):
    scrapeDataframe['reviews.cleantext'] = scrapeDataframe['reviews.text']
    scrapeDataframe = categoryReplace(scrapeDataframe)
    scrapeDataframe = removeStopwords(scrapeDataframe)
    #remove ... after sentence
    scrapeDataframe['reviews.cleantext'] = scrapeDataframe['reviews.cleantext'].apply(remove_sentence)    
    # Remove emoji symbols from the column
    scrapeDataframe['reviews.cleantext'] = scrapeDataframe['reviews.cleantext'].apply(lambda x: emoji_pattern.sub(r'', x))
    # Remove additonal stopwords
    scrapeDataframe['reviews.cleantext'] = scrapeDataframe['reviews.cleantext'].apply(process_review)

    #scrapeDataframe['reviews.title'] = scrapeDataframe['reviews.title'].apply(lambda x: emoji_pattern.sub(r'', x))
    #scrapeDataframe['reviews.text'] = scrapeDataframe['reviews.text'].apply(lambda x: emoji_pattern.sub(r'', x))

    scrapeDataframe.dropna(subset=['reviews.cleantext'], inplace=True)

    #Keep rows with English sentences
    #scrapeDataframe = scrapeDataframe[scrapeDataframe['reviews.title'].apply(is_english)]
    return scrapeDataframe


def readScrapeCsv(filename:str) -> pd.DataFrame:
    dateAdded, dateUpdated, address, categories, primaryCategories, city, country, keys, latitude, longitude, name, postalCode, province, reviews_date, reviews_dateSeen, reviews_rating, reviews_sourceURLs, reviews_text, reviews_title, reviews_userCity, reviews_userProvince, reviews_username, sourceURLs, websites = ([] for _ in range(24))
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
    'websites': websites
    }

    with open(filename, encoding='utf-8-sig' and 'utf-8') as f:
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
            reviews_rating.append(row[globalVar.REVIEWS_RATING])
            reviews_sourceURLs.append(row[globalVar.REVIEWS_SOURCEURLS])
            reviews_text.append(row[globalVar.REVIEWS_TEXT])
            reviews_title.append(row[globalVar.REVIEWS_TITLE])
            reviews_userCity.append(row[globalVar.REVIEWS_USERCITY])
            reviews_userProvince.append(row[globalVar.REVIEWS_USERPROVINCE])
            reviews_username.append(row[globalVar.REVIEWS_USERNAME])
            sourceURLs.append(row[globalVar.SOURCEURLS])
            websites.append(row[globalVar.WEBSITES])

    return pd.DataFrame(data)

def main():
    dataFrame = readScrapeCsv(r'C:\Users\anyho\Desktop\PythonProject\PythonApp\sample.csv')
    dataCleaning(dataFrame).to_csv(r"C:\Users\anyho\Desktop\PythonProject\PythonApp\sampleclean.csv")

main()





