import csv
from selenium import webdriver
import time
from selenium.webdriver.common.by import By

def get_reviews_data(driver, csvWriter):
    container = driver.find_elements(By.XPATH, "//div[@data-reviewid]")
    dates = driver.find_elements(By.XPATH, ".//div[@class='cRVSd']")
    hotel_name = driver.find_element(By.ID, 'HEADING').text
    location = driver.find_element(By.XPATH, ".//span[contains(@class, 'oAPmj _S')]").text

    for j in range(len(container)): #A loop defined by the number of reviews
        rating = container[j].find_element(By.XPATH, ".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        title = container[j].find_element(By.XPATH, ".//div[contains(@data-test-target, 'review-title')]").text
        review = container[j].find_element(By.XPATH, ".//span[@class='QewHA H4 _a']").text.replace("\n", "  ")
        date = " ".join(dates[j].text.split(" ")[-2:])
        csvWriter.writerow([hotel_name, date, rating, title, review, location])

def main():
    #Create a CSV file in your local host and place the directory to "path_to_file"
    path_to_file = (r"  ") ######################################Input path here########################################
    pages_to_scrape = 1000 #if there is 1000 reviews, divide by 10 to get number of pages
    url=input("Enter your Tripadvisor review if it has this in the URL '/Hotel_Review-g294265-d12825240-Reviews-' :")
    driver = webdriver.Chrome()
    driver.get(url)
    csvFile = open(path_to_file, 'a', encoding="utf-8")
    csvWriter = csv.writer(csvFile)

    for i in range(0, pages_to_scrape):
        time.sleep(5)
        driver.find_element(By.XPATH, ".//div[contains(@data-test-target, 'expand-review')]").click()
        get_reviews_data(driver, csvWriter)
        driver.find_element(By.XPATH, './/a[@class="ui_button nav next primary "]').click()

    driver.quit()

if __name__ == "__main__":
    main()