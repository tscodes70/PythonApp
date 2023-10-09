import dataScrape,dataClean,dataStandardization,dataAnalyze
import time

def main():
    sTime = time.time() 
    dataScrape.dataScraper()
    dataClean.dataCleaner()
    dataStandardization.handleMissingData()
    dataAnalyze.dataAnalysis()
    eTime = time.time() 

    runtime = eTime - sTime
    print(f"======= Runtime Information =======")
    print(f"Runtime: {runtime} seconds")
    print(f"===================================")
    
main()