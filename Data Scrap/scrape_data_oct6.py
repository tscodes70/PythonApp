import csv
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementNotInteractableException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

    if redirect_href is not None:
        href_value = redirect_href.get_attribute("href")
        
        if href_value:
            country = driver.find_element(By.XPATH, ".//li[contains(@class, 'breadcrumb')]").text
            categories = driver.find_element(By.XPATH, ".//div[contains(@class, 'cGAqf')]").text.split(" ")[3]

            prices = driver.find_elements(By.XPATH, ".//div[contains(@class, 'hhlrH w')]")
            sub_categories = driver.find_elements(By.XPATH, "//div[@class='OsCbb K']")
            combined_amenities = []
            combined_prices = []

            for sub_category in sub_categories:
                amenities_elements = sub_category.find_elements(By.XPATH, ".//div[contains(@data-test-target, 'amenity_text')]")
                amenities_text = ', '.join(amenity.text for amenity in amenities_elements)
                combined_amenities.append(amenities_text)

            for price in prices:
                price_elements = price.find_elements(By.XPATH, ".//div[contains(@data-sizegroup, 'hr_chevron_prices')]")
                price_texts = [price_element.text.strip() for price_element in price_elements]
                price_text = ', '.join(price_texts)
                combined_prices.append(price_text)

            cleaned_prices = ', '.join(combined_prices)
            cleaned_prices = cleaned_prices.replace("[", "").replace("]", "").replace("'", "")

            for j in range(len(container)):
                rating = container[j].find_element(By.XPATH, ".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
                title = container[j].find_element(By.XPATH, ".//div[contains(@data-test-target, 'review-title')]").text
                review = container[j].find_element(By.XPATH, ".//span[@class='QewHA H4 _a']").text.replace("\n", "  ")
                date = " ".join(dates[j].text.split(" ")[-2:])
                # Wait for username to be visible
                #username_element = WebDriverWait(driver, 10).until(
                    #EC.visibility_of_element_located((By.XPATH, ".//a[@class='ui_header_link uyyBf']"))
                #)
                #username = username_element.text

                csvWriter.writerow([address, categories, city, country, hotel_name, postal_code, province, date, rating, title, review, '''username''', source_url, href_value, combined_amenities, cleaned_prices])
       
def main():
    path_to_file = r"C:\Users\Nisa\OneDrive - Temasek Polytechnic\Documents\Singapore Institute of Technology\Y1 TRIMESTER 1\INF1002 Programming fundamental\Project\test_5oct.csv"
    url = input("Enter your Tripadvisor review URL: ")
    driver = webdriver.Chrome()
    driver.get(url)
    csvFile = open(path_to_file, 'a', encoding="utf-8")
    csvWriter = csv.writer(csvFile)

    # Write header only once
    if driver.current_url == url:
        csvWriter.writerow(["address", "categories", "city",  "country", "hotelName", "postalCode", "province", "reviews.date", "reviews.rating", "reviews.title", "reviews.text", "reviews.username", "reviews.sourceURLs", "sourceURL", "amenities", "prices"])

    while True:
        time.sleep(5)
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

if __name__ == "__main__":
    main()
