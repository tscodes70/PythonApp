import csv, re, nltk
import pandas as pd
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
import globalVar

CUSTOMSTOPWORDS = r'stopword_list.txt'

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


# Define function to ... after sentence
def remove_sentence(sentence):
    # Match regex pattern
    sentence_pattern = r'\.\.\.$'

    # Use re.sub to replace the sentence pattern with an empty string
    cleaned_sentence = re.sub(sentence_pattern, '', sentence)

    return cleaned_sentence


# Return true if its English, else false
def is_english(sentence):
    try:
        return detect(sentence) == 'en'
    except Exception as e:
        print(f"Error detecting language for '{sentence}': {str(e)}")
        return False


def categoryReplace(dataFrame: pd.DataFrame) -> pd.DataFrame:
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


def removeStopwords(dataFrame: pd.DataFrame):
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

# Define a function to clean the amenities
def clean_amenities(amenities):
    if amenities is not None and isinstance(amenities, str):
        # Split the amenities by commas and remove empty entries
        cleaned_amenities = [amenity.strip() for amenity in amenities.split(',') if amenity.strip()]
        # If there are no cleaned amenities, set it to an empty list
        if not cleaned_amenities:
            return '[]'
        # Join the cleaned amenities back into a single string
        return ', '.join(cleaned_amenities)
    else:
        return '[]'  # Return an empty list if it's not a valid string

# Define a function to calculate the average of three values in a row and replace the 'prices' column
def calculate_average_and_replace(row):
    if 'prices' in row and row['prices'] is not None and row['prices'] != '':
        prices = [int(price.strip('S$')) for price in row['prices'].split(', ') if price.strip('S$').isdigit()]
        average_price = sum(prices) / len(prices)
        # Replace the 'prices' column with the average
        row['prices'] = f'{average_price:.2f}'

    return row

# Function to replace 'United States' with 'US' using regular expressions
def replace_country_name(country_name):
    pattern = re.compile(r'United\s*States', re.IGNORECASE)
    if pattern.search(country_name):
        return 'US'
    return country_name

# Define a function to convert date strings to the desired format
def custom_date_parser(date_string):
    # Regular expressions to match different date formats
    date_string = date_string.replace("Sept", "Sep")
    regex_formats = [
        (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$', '%Y-%m-%dT%H:%M:%SZ'),  # Already in the correct format
        (r'^\d{2}-[A-Z][a-z]{2}-\d{2}$', '%d-%b-%y'),  # e.g., 5-Oct-23
        (r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}$', '%b-%y'),  # e.g., Jul-14
        (r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}$", '%b %Y')  # e.g., Jul 2014
    ]
    regex_formats2 = [
        (r"^\d{1} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$"),  # e.g., 1 Jul
        (r"^\d{2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$")  # e.g., 10 Jul
    ]

    # Loop through regex formats and try to match
    for regex, format_str in regex_formats:
        if re.match(regex, date_string):
            return str(pd.to_datetime(date_string, format=format_str).strftime('%Y-%m-%dT00:00:00Z'))
    
    for regex in regex_formats2: 
        if re.match(regex, date_string):
            parsed_date = pd.to_datetime(date_string, format='%d %b')
            current_year = pd.Timestamp.now().year
            return f'{current_year:04d}-{parsed_date.month:02d}-{parsed_date.day:02d}T00:00:00Z'

    if date_string == "review Yesterday":
        current_day = pd.Timestamp.now().day
        current_month = pd.Timestamp.now().month
        current_year = pd.Timestamp.now().year
        return f'{current_year:04d}-{current_month:02d}-{int(current_day)-1:02d}T00:00:00Z'
    # For the "5-Oct" format, parse it differently
    try:
        parsed_date = pd.to_datetime(date_string, format='%d-%b')
        current_year = pd.Timestamp.now().year
        return f'{current_year:04d}-{parsed_date.month:02d}-{parsed_date.day:02d}T00:00:00Z'
    except ValueError:
        pass

    # Return the original date string if no match is found
    return date_string

def dataCleaning(scrapeDataframe: pd.DataFrame):
    scrapeDataframe['reviews.cleantext'] = scrapeDataframe['reviews.text']
    scrapeDataframe = categoryReplace(scrapeDataframe)
    scrapeDataframe = removeStopwords(scrapeDataframe)
    # remove ... after sentence
    scrapeDataframe['reviews.cleantext'] = scrapeDataframe['reviews.cleantext'].apply(remove_sentence)
    # Remove emoji symbols from the column
    scrapeDataframe['reviews.cleantext'] = scrapeDataframe['reviews.cleantext'].apply(
        lambda x: emoji_pattern.sub(r'', x))
    # Remove additonal stopwords
    scrapeDataframe['reviews.cleantext'] = scrapeDataframe['reviews.cleantext'].apply(process_review)

    # scrapeDataframe['reviews.title'] = scrapeDataframe['reviews.title'].apply(lambda x: emoji_pattern.sub(r'', x))
    # scrapeDataframe['reviews.text'] = scrapeDataframe['reviews.text'].apply(lambda x: emoji_pattern.sub(r'', x))

    scrapeDataframe.dropna(subset=['reviews.cleantext'], inplace=True)

    # Keep rows with English sentences
    # scrapeDataframe = scrapeDataframe[scrapeDataframe['reviews.title'].apply(is_english)]

    # Apply the clean_amenities function to the 'amenities' column
    try:
        if 'amenities' in scrapeDataframe.columns:
            scrapeDataframe['amenities'] = scrapeDataframe['amenities'].apply(clean_amenities)
        else:
            print("The 'amenities' column does not exist in the DataFrame.")
    except KeyError as e:
        print(f"KeyError: {str(e)}")
        # Calculate the average prices and replace them
    scrapeDataframe = scrapeDataframe.apply(calculate_average_and_replace, axis=1)

    try:
        scrapeDataframe['reviews.rating'] = pd.to_numeric(scrapeDataframe['reviews.rating'], errors='coerce')
        scrapeDataframe['reviews.rating'] = scrapeDataframe['reviews.rating'].apply(lambda x: x // 10 if not pd.isnull(x) and x >= 10 else x)
    except Exception as e:
        print(f"Error converting and updating 'reviews.rating': {str(e)}")

    # Apply the replacement function to the 'country' column
    scrapeDataframe['country'] = scrapeDataframe['country'].apply(replace_country_name)

    # Apply the custom date parsing function to the 'reviews.date' column
    scrapeDataframe['reviews.date'] = scrapeDataframe['reviews.date'].apply(custom_date_parser)

    # Now, 'reviews.date' contains the transformed dates

    return scrapeDataframe

def split_amenities(row):
    return row[globalVar.AMENITIES].split(', ')

def remove_empty_strings(lst):
    return [item for item in lst if item.strip() != '']


def readScrapeCsv(filename: str) -> pd.DataFrame:
    dateAdded, dateUpdated, address, categories, primaryCategories, city, country, keys, latitude, longitude, name, postalCode, province, reviews_date, reviews_dateSeen, reviews_rating, reviews_sourceURLs, reviews_text, reviews_title, reviews_userCity, reviews_userProvince, reviews_username, sourceURLs, websites, amenities, prices  = (
    [] for _ in range(26))
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
        'amenities': amenities,
        'prices': prices
    }

    with open(filename, encoding='utf-8-sig' and 'utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['dateAdded'].append(row.get(globalVar.DATEADDED))
            data['dateUpdated'].append(row.get(globalVar.DATEUPDATED))
            data['address'].append(row.get(globalVar.ADDRESS))
            data['categories'].append(row.get(globalVar.CATEGORIES))
            data['primaryCategories'].append(row.get(globalVar.PRIMARYCATEGORIES))
            data['city'].append(row.get(globalVar.CITY))
            data['country'].append(row.get(globalVar.COUNTRY))
            data['keys'].append(row.get(globalVar.KEYS))
            data['latitude'].append(row.get(globalVar.LATITUDE))
            data['longitude'].append(row.get(globalVar.LONGITUDE))
            data['name'].append(row.get(globalVar.NAME))
            data['postalCode'].append(row.get(globalVar.POSTALCODE))
            data['province'].append(row.get(globalVar.PROVINCE))
            data['reviews.date'].append(row.get(globalVar.REVIEWS_DATE))
            data['reviews.dateSeen'].append(row.get(globalVar.REVIEWS_DATESEEN))
            data['reviews.rating'].append(row.get(globalVar.REVIEWS_RATING))
            data['reviews.sourceURLs'].append(row.get(globalVar.REVIEWS_SOURCEURLS))
            data['reviews.text'].append(row.get(globalVar.REVIEWS_TEXT))
            data['reviews.title'].append(row.get(globalVar.REVIEWS_TITLE))
            data['reviews.userCity'].append(row.get(globalVar.REVIEWS_USERCITY))
            data['reviews.userProvince'].append(row.get(globalVar.REVIEWS_USERPROVINCE))
            data['reviews.username'].append(row.get(globalVar.REVIEWS_USERNAME))
            data['sourceURLs'].append(row.get(globalVar.SOURCEURLS))
            data['websites'].append(row.get(globalVar.WEBSITES))
            data['amenities'].append(row.get(globalVar.AMENITIES))
            data['prices'].append(row.get(globalVar.PRICES))

    return pd.DataFrame(data)


def dataCleaner(INPUTFULLFILE,OUTPUTFULLFILE):
    dataFrame = readScrapeCsv(INPUTFULLFILE)
    cleanedDataFrame = dataCleaning(dataFrame)
    cleanedDataFrame[globalVar.AMENITIES] = cleanedDataFrame[globalVar.AMENITIES].str.replace("[", "").str.replace("]", "").str.replace("'", "")
    cleanedDataFrame[globalVar.AMENITIES] = cleanedDataFrame.apply(split_amenities, axis=1)
    cleanedDataFrame[globalVar.AMENITIES] = cleanedDataFrame[globalVar.AMENITIES].apply(remove_empty_strings)


    # Check if the dataclean.csv file exists
    try:
        existing_data = pd.read_csv(OUTPUTFULLFILE)
        # Append the cleaned data to the existing data
        combined_data = pd.concat([existing_data, cleanedDataFrame], ignore_index=True)
        # Save the combined data to the dataclean.csv file
        combined_data.to_csv(OUTPUTFULLFILE, index=False)
        print(f"Data appended to {OUTPUTFULLFILE}")
    except FileNotFoundError:
        # If the file doesn't exist, save the cleaned data directly
        cleanedDataFrame.to_csv(OUTPUTFULLFILE, index=False)
        print(f"Data saved to {OUTPUTFULLFILE}")


