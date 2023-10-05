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
    #redirect_href=driver.find_element(By.XPATH, ".//a[contains(@href)]")
    country = driver.find_element(By.XPATH, ".//li[contains(@class, 'breadcrumb')]").text
    categories=driver.find_element(By.XPATH,".//div[contains(@class, 'cGAqf')]").text.split(" ")[3]
 
    
    sub_categories = driver.find_elements(By.XPATH, "//div[@class='OsCbb K']")
    combined_amenities = []

    for sub_category in sub_categories:
        amenities_elements = sub_category.find_elements(By.XPATH, ".//div[contains(@data-test-target, 'amenity_text')]")
        amenities_text = ', '.join(amenity.text for amenity in amenities_elements)
        combined_amenities.append(amenities_text)
 
    
    
    for j in range(len(container)): #A loop defined by the number of reviews
        rating = container[j].find_element(By.XPATH, ".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        title = container[j].find_element(By.XPATH, ".//div[contains(@data-test-target, 'review-title')]").text
        review = container[j].find_element(By.XPATH, ".//span[@class='QewHA H4 _a']").text.replace("\n", "  ")
        date = " ".join(dates[j].text.split(" ")[-2:])
        #Wait for username to be visible
        username_element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.XPATH, ".//a[@class='ui_header_link uyyBf']"))
        )
        username = username_element.text
        
        csvWriter.writerow([address, categories, city, country, hotel_name, postal_code, province, date, rating, title, review, username, source_url, '''redirect_href''', combined_amenities])


def main():
    path_to_file = (r"C:\Users\Nisa\OneDrive - Temasek Polytechnic\Documents\Singapore Institute of Technology\Y1 TRIMESTER 1\INF1002 Programming fundamental\Project\test_5oct.csv")
    url = input("Enter your Tripadvisor review URL: ")
    driver = webdriver.Chrome()
    driver.get(url)
    csvFile = open(path_to_file, 'a', encoding="utf-8")
    csvWriter = csv.writer(csvFile)

    # Write header only once
    if driver.current_url == url:
        csvWriter.writerow(["address", "categories", "city",  "country", "hotelName", "postalCode", "province", "reviews.date", "reviews.rating", "reviews.title", "reviews.text", "reviews.username", "reviews.sourceURLs", "sourceURL", "amenities"])

    while True:
        time.sleep(5)
        try:
            expand_review = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, ".//div[contains(@data-test-target, 'expand-review')]"))
            )
            expand_review.click()
            get_reviews_data(driver, csvWriter)
        except (ElementNotInteractableException,NoSuchElementException):
            break

        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, './/a[@class="ui_button nav next primary "]'))
            )
            next_button.click()
        except (NoSuchElementException,TimeoutException):
            break

    driver.quit()

if __name__ == "__main__":
    main()
