#This is a testing web scraper

import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://data.world/datafiniti/hotel-reviews"

# GET request
response = requests.get(URL)

# Check if page reachable
if response.status_code == 200:
    # Parse html
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract data
    links = soup.find_all('a')

    data = {'Link Text': [link.text for link in links],
            'Link URL': [link.get('href') for link in links]}
    df = pd.DataFrame(data)
    print(df)
else:
    print("Error fetching page. Status code:", response.status_code)
