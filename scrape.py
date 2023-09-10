import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL of the web page you want to scrape
url = "https://data.world/datafiniti/hotel-reviews"

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find and extract specific elements from the page (e.g., links)
    links = soup.find_all('a')


    # Create a pandas DataFrame to store the extracted data
    data = {'Link Text': [link.text for link in links],
            'Link URL': [link.get('href') for link in links]}

    
    df = pd.DataFrame(data)

    # Print the DataFrame
    print(df)
else:
    print("Failed to retrieve the web page. Status code:", response.status_code)
