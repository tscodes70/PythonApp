# General CSV Headers
ID = 'id'
DATEADDED = 'dateAdded'
DATEUPDATED = 'dateUpdated'
ADDRESS = 'address'
CATEGORIES = 'categories'
PRIMARYCATEGORIES = 'primaryCategories'
CITY = 'city'
COUNTRY = 'country'
KEYS = 'keys'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
NAME = 'name'
POSTALCODE = 'postalCode'
PROVINCE = 'province'
REVIEWS_DATE = 'reviews.date'
REVIEWS_DATESEEN = 'reviews.dateSeen'
REVIEWS_RATING = 'reviews.rating'
REVIEWS_SOURCEURLS = 'reviews.sourceURLs'
REVIEWS_TEXT = 'reviews.text'
REVIEWS_TITLE = 'reviews.title'
REVIEWS_USERCITY = 'reviews.userCity'
REVIEWS_USERPROVINCE = 'reviews.userProvince'
REVIEWS_USERNAME = 'reviews.username'
SOURCEURLS = 'sourceURLs'
WEBSITES = 'websites'


# Crawling Variables

# Cleaning Variables

# Analysis Variables
REVIEWS_CLEANTEXT = 'reviews.cleantext'
REVIEWS_SUMMARY = 'reviews.summary'
REVIEWS_TOTAL = 'reviews.total'
AVERAGE_RATING = 'average.rating'
POPULAR_KEYWORDS = "popular.keywords"

SENTIMENT_SCORE = 'Sentiment Score'
COMPOUND_SENTIMENT_SCORE = 'Compound Sentiment'

RATINGMAX = 5
KEYWORDMAX = 10
REVIEWSUMMAX = 2

ANALYSISENCODING = "utf8"


ANALYSISINPUTFILE = r"C:\Users\anyho\Desktop\PythonProject\csvs\testing.csv"
ANALYSISOUTPUTHEADER = [NAME, PROVINCE, POSTALCODE,CATEGORIES,PRIMARYCATEGORIES, COMPOUND_SENTIMENT_SCORE, REVIEWS_SUMMARY, REVIEWS_TOTAL, POPULAR_KEYWORDS,AVERAGE_RATING]
ANALYSISOUTPUTBYHOTEL = 'Analysis_GHotel.csv'
ANALYSISOUTPUTBYREVIEWS = 'Analysis_Reviews.csv'
