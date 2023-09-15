import requests
from bs4 import BeautifulSoup

# URL of the Trivago search results page you want to scrape
url = "https://www.priceline.com/relax/in/3000040021/from/20231011/to/20231013/rooms/1"

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find and extract specific elements from the page
    # hotel_names = soup.find_all("h5", class_="name__copytext")
    # hotel_prices = soup.find_all("span", class_="esgW-price")

    with open('scrape.txt', 'w') as f:
        f.write(str(soup.find('html')))
    # Print the extracted data
    # for name, price in zip(hotel_names, hotel_prices):
    #     print("Hotel Name:", name.get_text())
    #     print("Price:", price.get_text())
    #     print("\n")
else:
    print("Failed to retrieve the page. Status code:", response.status_code)
