# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:16:49 2023

@author: Nisa
"""

import csv
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

def get_reviews_data(driver, csvWriter):
    container = driver.find_elements(By.XPATH, "//div[@data-reviewid]")
    dates = driver.find_elements(By.XPATH, ".//div[@class='cRVSd']")
    hotel_name = driver.find_element(By.ID, 'HEADING').text
    full_location = driver.find_element(By.XPATH, ".//span[contains(@class, 'oAPmj _S')]").text
    postal_province_data = driver.find_element(By.XPATH, ".//span[contains(@class, 'oAPmj _S')]").text
    postal_province_parts = postal_province_data
    postal_code = postal_province_parts.split(" ")[-1]
    province = postal_province_parts.split(" ")[-2]
    city = postal_province_parts.split(",")[-2]
    source_url = driver.current_url
    #redirect_href=driver.find_element(By.XPATH, ".//a[contains(@href)]")
    country = driver.find_element(By.XPATH, ".//li[contains(@class, 'breadcrumb')]").text
    categories=driver.find_element(By.XPATH,".//div[contains(@class, 'cGAqf')]").text.split(" ")[3]
    
    
    for j in range(len(container)): #A loop defined by the number of reviews
        rating = container[j].find_element(By.XPATH, ".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        title = container[j].find_element(By.XPATH, ".//div[contains(@data-test-target, 'review-title')]").text
        review = container[j].find_element(By.XPATH, ".//span[@class='QewHA H4 _a']").text.replace("\n", "  ")
        date = " ".join(dates[j].text.split(" ")[-2:])
        csvWriter.writerow([full_location, country, categories, city, hotel_name, postal_code, province, date, rating, title, review, source_url, '''redirect_href'''])



def main():
    #Create a CSV file in your local host and place the directory to "path_to_file"
    path_to_file = (r"C:\Users\Nisa\OneDrive - Temasek Polytechnic\Documents\Singapore Institute of Technology\Y1 TRIMESTER 1\INF1002 Programming fundamental\Project\test_4oct.csv") ######################################Input path here########################################
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
        try:
            driver.find_element(By.XPATH, './/a[@class="ui_button nav next primary "]').click()
        except NoSuchElementException:
            break
    
    driver.quit()


if __name__ == "__main__":
    main()