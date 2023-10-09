# -*- coding: utf-8 -*-
"""
Created on Sat Oct 7 00:11:08 2023

@author: Timothy
"""
import os
from pathlib import Path
import datetime
CWD = os.path.dirname(__file__)
PCWD = os.path.join(CWD, "..")
CSVD = os.path.join(PCWD, "csvs")

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

PRICES = 'prices'
AMENITIES = 'amenities'

# Crawling Variables
CURRENTDATE = datetime.date.today().strftime("%d-%b")
CRAWLEROUTPUTFILE = f"crawl_{CURRENTDATE}.csv"
CRAWLEROUTPUTFULLFILE = os.path.join(CSVD,CRAWLEROUTPUTFILE)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
SCRAPE_URLS = (
"https://www.tripadvisor.com.sg/Hotel_Review-g34438-d23444061-Reviews-or210-YOTEL_Miami-Miami_Florida.html#REVIEWS," +
"https://www.tripadvisor.com.sg/Hotel_Review-g34438-d25150899-Reviews-The_Elser_Hotel_Residences-Miami_Florida.html," +
"https://www.tripadvisor.com.sg/Hotel_Review-g46686-d279807-Reviews-Florentine_Family_Motel-North_Wildwood_Cape_May_County_New_Jersey.html," +
"https://www.tripadvisor.com.sg/Hotel_Review-g46341-d1205812-Reviews-Blue_Fish_Inn-Cape_May_Cape_May_County_New_Jersey.html," +
"https://www.tripadvisor.com.sg/Hotel_Review-g60750-d80602-Reviews-Town_and_Country_Resort-San_Diego_California.html" +
"https://www.tripadvisor.com.sg/Hotel_Review-g29092-d75532-Reviews-The_Anaheim_Hotel-Anaheim_California.html," +
"https://www.tripadvisor.com.sg/Hotel_Review-g34439-d23093311-Reviews-The_Goodtime_Hotel-Miami_Beach_Florida.html," +
"https://www.tripadvisor.com.sg/Hotel_Review-g60742-d14143093-Reviews-Hampton_Inn_Suites_Asheville_Biltmore_Area-Asheville_North_Carolina.html," +
"https://www.tripadvisor.com.sg/Hotel_Review-g49022-d19414709-Reviews-Grand_Bohemian_Hotel_Charlotte_Autograph_Collection-Charlotte_North_Carolina.html"
)

# Cleaning Variables
CLEANERINPUTFULLFILE = CRAWLEROUTPUTFULLFILE
CLEANEROUTPUTFILE = f"clean_{CURRENTDATE}.csv"
CLEANEROUTPUTFULLFILE = os.path.join(CSVD,CLEANEROUTPUTFILE)

# Missing Data Variables
MDINPUTFULLFILE = CLEANEROUTPUTFULLFILE
MDOUTPUTFILE = f"missingdata_{CURRENTDATE}.csv"
MDOUTPUTFULLFILE = os.path.join(CSVD,MDOUTPUTFILE)

# Analysis Variables
ANALYSISINPUTFULLFILE = MDINPUTFULLFILE
ANALYSISREVIEWOUTPUTFILE = f"analyzedreviews_{CURRENTDATE}.csv"
ANALYSISHOTELOUTPUTFILE = f"analyzedhotels_{CURRENTDATE}.csv"
ANALYSISREVIEWOUTPUTFULLFILE = os.path.join(CSVD,ANALYSISREVIEWOUTPUTFILE)
ANALYSISHOTELOUTPUTFULLFILE = os.path.join(CSVD,ANALYSISHOTELOUTPUTFILE)

ANALYSISENCODING = "utf8"

REVIEWS_CLEANTEXT = 'reviews.cleantext'
REVIEWS_SUMMARY = 'reviews.summary'
REVIEWS_TOTAL = 'reviews.total'
REVIEWS_LENGTH = 'reviews.length'
AVERAGE_REVIEWS_LENGTH = 'average.reviews.length'
AVERAGE_RATING = 'average.rating'
POPULAR_KEYWORDS = "popular.keywords"

SENTIMENT_SCORE = 'Sentiment Score'
COMPOUND_SENTIMENT_SCORE = 'Compound Sentiment'

RATINGMAX = 5
KEYWORDMAX = 10
REVIEWSUMMAX = 2

ANALYSISOUTPUTHEADER = [NAME, PROVINCE, POSTALCODE,CATEGORIES,PRIMARYCATEGORIES, COMPOUND_SENTIMENT_SCORE, REVIEWS_SUMMARY, REVIEWS_TOTAL, POPULAR_KEYWORDS,AVERAGE_RATING]


