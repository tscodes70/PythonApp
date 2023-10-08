# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:50:45 2023

@author: Nisa
"""

import csv
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementNotInteractableException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Define a function to get prices
def get_prices(driver):
    try:
        # Try to find the first price structure element
        price_elements = driver.find_elements(By.XPATH, ".//div[contains(@class, 'hhlrH w')]//div[contains(@data-sizegroup, 'hr_chevron_prices')]")
        combined_prices = []

        for price in price_elements:
            price_text = price.text.strip()
            combined_prices.append(price_text)

        if combined_prices:
            cleaned_prices = ', '.join(combined_prices)
        else:
            # If the first structure is not found, try to find the second structure
            price_elements2 = driver.find_elements(By.XPATH, ".//div[contains(@class, 'JPNOn JPNOn')]")
            cleaned_prices = ', '.join(price.text.strip() for price in price_elements2)

        cleaned_prices = cleaned_prices.replace("[", "").replace("]", "").replace("'", "")
    except NoSuchElementException:
        # If neither structure is found, set cleaned_prices to "NA"
        cleaned_prices = "NA"
    
    return cleaned_prices





def header_exists(file_path):
    try:
        with open(file_path, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            return header == ["address", "categories", "city",  "country", "hotelName", "postalCode", "province", "reviews.date", "reviews.rating", "reviews.title", "reviews.text", "reviews.sourceURLs", "sourceURL", "amenities", "prices"]
    except FileNotFoundError:
        return False





def get_reviews_data(driver, csvWriter):
    container = driver.find_elements(By.XPATH, "//div[@data-reviewid]")
    dates = driver.find_elements(By.XPATH, ".//div[@class='cRVSd']")
    hotel_name = driver.find_element(By.ID, 'HEADING').text
    full_location = driver.find_element(By.XPATH, ".//span[contains(@class, 'oAPmj _S')]").text
    location_parts = full_location
    address = location_parts.split(",")[0]
    postal_code = location_parts.split(" ")[-1]
    province = location_parts.split(" ")[-2]
    city = location_parts.split(",")[-2]
    source_url = driver.current_url

    # Find redirect_href element
    redirect_href = None
    try:
        redirect_href = driver.find_element(By.XPATH, ".//a[contains(@class, 'YnKZo Ci Wc _S C pInXB _S ITocq jNmfd')]")
    except NoSuchElementException:
        pass

    href_value = redirect_href.get_attribute("href") if redirect_href else None

    country = driver.find_element(By.XPATH, ".//li[contains(@class, 'breadcrumb')]").text
    categories = driver.find_element(By.XPATH, ".//div[contains(@class, 'cGAqf')]").text.split(" ")[3]
    sub_categories = driver.find_elements(By.XPATH, "//div[@class='OsCbb K']")
    combined_amenities = []
    cleaned_prices = get_prices(driver)

    for sub_category in sub_categories:
        amenities_elements = sub_category.find_elements(By.XPATH, ".//div[contains(@data-test-target, 'amenity_text')]")
        amenities_text = ', '.join(amenity.text for amenity in amenities_elements)
        combined_amenities.append(amenities_text)

    for j in range(len(container)):
        rating = container[j].find_element(By.XPATH, ".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        rating_one_digit = rating[:1]
        title = container[j].find_element(By.XPATH, ".//div[contains(@data-test-target, 'review-title')]").text
        review = container[j].find_element(By.XPATH, ".//span[@class='QewHA H4 _a']").text.replace("\n", "  ")
        date = " ".join(dates[j].text.split(" ")[-2:])

        csvWriter.writerow([address, categories, city, country, hotel_name, postal_code, province, date, rating_one_digit, title, review, source_url, href_value, combined_amenities, cleaned_prices])




def main():
    path_to_file = r"C:\Users\Nisa\OneDrive - Temasek Polytechnic\Documents\Singapore Institute of Technology\Y1 TRIMESTER 1\INF1002 Programming fundamental\Project\test_5oct.csv"
    urls = input("Enter TripAdvisor review URLs (comma-separated): ").split(',')
    # Define the custom user-agent
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    # Configure Chrome WebDriver with custom user-agent
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-agent={user_agent}")
    driver = webdriver.Chrome()
    csvFile = open(path_to_file, 'a', encoding="utf-8", newline='')

    # Check if header exists and write it if not
    if not header_exists(path_to_file):
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["address", "categories", "city",  "country", "hotelName", "postalCode", "province", "reviews.date", "reviews.rating", "reviews.title", "reviews.text", "reviews.sourceURLs", "sourceURL", "amenities", "prices"])

    for url in urls:
        driver.get(url)
        csvWriter = csv.writer(csvFile)
        
        while True:
            time.sleep(6)
            try:
                expand_review = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, ".//div[contains(@data-test-target, 'expand-review')]"))
                )
                expand_review.click()
                get_reviews_data(driver, csvWriter)
            except (ElementNotInteractableException, NoSuchElementException, TimeoutException):
                break

            try:
                next_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, './/a[@class="ui_button nav next primary "]'))
                )
                next_button.click()
            except (NoSuchElementException, TimeoutException):
                break

    driver.quit()
    csvFile.close()

if __name__ == "__main__":
    main()
