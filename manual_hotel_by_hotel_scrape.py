# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:44:27 2023

@author: Nisa
"""
import csv 
from selenium import webdriver 
import time
from selenium.webdriver.common.by import By
#from selenium.webdriver.common.keys import Keys
#Create a csv in your local comp and direct a directory there
path_to_file = (r"C:\Users\Nisa\OneDrive - Temasek Polytechnic\Documents\Singapore Institute of Technology\Y1 TRIMESTER 1\INF1002 Programming fundamental\Project\Tripadvisor_Singapore_10hotels_part2.csv")
#Will fix this to make it go to the end of review page
pages_to_scrape = 320
url="https://www.tripadvisor.com/Hotel_Review-g294265-d19888117-Reviews-The_Clan_Hotel_by_Far_East_Hospitality-Singapore.html"
driver = webdriver.Chrome()
driver.get(url)
csvFile = open(path_to_file, 'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)
for i in range(0, pages_to_scrape):
    time.sleep(6)
    # Click the "expand review" link to reveal the entire review.
    driver.find_element(By.XPATH, ".//div[contains(@data-test-target, 'expand-review')]").click()
    container = driver.find_elements(By.XPATH,"//div[@data-reviewid]")
    dates = driver.find_elements(By.XPATH,".//div[@class='cRVSd']")
    hotel_name=driver.find_element(By.ID,'HEADING').text
    location=driver.find_element(By.XPATH,".//span[contains(@class, 'oAPmj _S')]").text
    
    for j in range(len(container)): # A loop defined by the number of reviews
        # Grab the rating and hotel name to categorise
        #driver.find_element(by=By.TAG_NAME,value="body").send_keys(Keys.PAGE_DOWN)
        #time.sleep(1)
        rating = container[j].find_element(By.XPATH,".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        title = container[j].find_element(By.XPATH,".//div[contains(@data-test-target, 'review-title')]").text
        review = container[j].find_element(By.XPATH,".//span[@class='QewHA H4 _a']").text.replace("\n", "  ")
        date = " ".join(dates[j].text.split(" ")[-2:])
        #Save that data in the csv
        csvWriter.writerow([hotel_name,date,rating, title, review,location])

    # After a page of review is done, repeat process for the next page
    driver.find_element(By.XPATH,'.//a[@class="ui_button nav next primary "]').click()
driver.quit()