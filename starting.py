import dataClean,dataStandardization,dataAnalyze
import time,traceback

# def fullFlow():
#     sTime = time.time() 
#     dataScrape.dataScraper()
#     dataClean.dataCleaner()
#     dataStandardization.handleMissingData()
#     dataAnalyze.dataAnalysis()
#     eTime = time.time() 

#     runtime = eTime - sTime
#     print(f"======= Runtime Information =======")
#     print(f"Runtime: {runtime} seconds")
#     print(f"===================================")

def customFileMain():
    sTime = time.time()
    dataClean.dataCleaner()
    dataStandardization.handleMissingData()
    dataAnalyze.dataAnalysis()
    eTime = time.time()

    runtime = eTime - sTime
    print(f"======= Runtime Information =======")
    print(f"Runtime: {runtime} seconds")
    print(f"===================================")

try:
    customFileMain()

except:
    traceback.print_exc() 