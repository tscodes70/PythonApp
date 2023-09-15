import csv 
from selenium import webdriver 
import time
from selenium.webdriver.common.by import By
#Create a csv in your local comp and direct a directory there
path_to_file = (r"C:\Users\Nisa\OneDrive - Temasek Polytechnic\Documents\Testing.csv")
#Will fix this to make it go to the end of review page
pages_to_scrape = 3 
url = "https://www.tripadvisor.com/Hotel_Review-g60982-d12077161-Reviews-Holiday_Inn_Express_Waikiki_an_IHG_Hotel-Honolulu_Oahu_Hawaii.html?spAttributionToken=MTg0OTA0MDY"
driver = webdriver.Chrome()
driver.get(url)
csvFile = open(path_to_file, 'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)
# change the value inside the range to save the number of reviews we're going to grab
for i in range(0, pages_to_scrape):
    time.sleep(3)
    # Click the "expand review" link to reveal the entire review.
    driver.find_element(By.XPATH, ".//div[contains(@data-test-target, 'expand-review')]").click()
    container = driver.find_elements(By.XPATH,"//div[@data-reviewid]")
    #dates = driver.find_elements(By.CLASS_NAME,"ui_header_link uyyBf")

   # Now we'll look at the reviews in the container and parse them out
    for j in range(len(container)): # A loop defined by the number of reviews

        # Grab the rating
        rating = container[j].find_element(By.XPATH,".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        title = container[j].find_element(By.XPATH,".//div[contains(@data-test-target, 'review-title')]").text
        review = container[j].find_element(By.XPATH,".//span[@class='QewHA H4 _a']").text.replace("\n", "  ")
        '''Code date is not working'''
        #date = driver.find_elements(By.CLASS_NAME,"ui_header_link uyyBf").get_attribute("class").split("_")[3]

        #Save that data in the csv
        csvWriter.writerow([rating, title, review])

    # After a page of review is done, repeat process for the next page
    driver.find_element(By.XPATH,'.//a[@class="ui_button nav next primary "]').click()
driver.quit()
