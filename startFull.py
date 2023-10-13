import dataAnalyze,timeAnalyze
import time,traceback,globalVar,os,datetime

def fullFlow():
    cin = globalVar.CLEANERINPUTFULLFILE
    cdfin = globalVar.CLEANERCUSTOMFULLFILE
    cout = globalVar.CLEANEROUTPUTFULLFILE
    mdin = globalVar.MDINPUTFULLFILE
    mdout = globalVar.MDOUTPUTFULLFILE
    ain = globalVar.ANALYSISINPUTFULLFILE
    arout = globalVar.ANALYSISREVIEWOUTPUTFULLFILE
    ahout = globalVar.ANALYSISHOTELOUTPUTFULLFILE
    getCorrelations = True

    sTime = time.time() 
    # dataScrape.dataScraper()
    # dataClean.dataCleaner(cin,cout)
    # dataClean.dataCleaner(cdfin,cout)
    # dataStandardization.handleMissingData(mdin,mdout)
    dataAnalyze.dataAnalysis(ain,arout,ahout,getCorrelations)
    timeAnalyze.timeAnalysis()
    eTime = time.time() 

    runtime = eTime - sTime
    print(f"======= Runtime Information =======")
    print(f"Runtime: {runtime} seconds")
    print(f"===================================")
try:
    fullFlow()

except:
    traceback.print_exc() 