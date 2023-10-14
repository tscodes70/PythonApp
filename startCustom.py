import dataClean,dataStandardization,dataAnalyze
import time,traceback,globalVar,os,datetime

def customFileMain(hoteldata:str):
    sTime = time.time()
    hotelfile = hoteldata[:-4]
    hotelclean = f"{hotelfile}_clean.csv"
    hotelmd = f"{hotelfile}_md.csv"
    hotelar = f"{hotelfile}_analyzedreviews.csv"
    hotelah = f"{hotelfile}_analyzedhotels.csv"

        
    fullhoteldata = os.path.join(globalVar.CSVD,hoteldata)
    fullhotelclean = os.path.join(globalVar.CSVD,hotelclean)
    fullhotelmd = os.path.join(globalVar.CSVD,hotelmd)
    fullhotelar = os.path.join(globalVar.CSVD,hotelar)
    fullhotelah = os.path.join(globalVar.CSVD,hotelah)
    getCorrelations = False

    dataClean.dataCleaner(fullhoteldata,fullhotelclean)
    dataStandardization.handleMissingData(fullhotelclean,fullhotelmd)
    dataAnalyze.dataAnalysis(fullhotelmd,fullhotelar,fullhotelah,getCorrelations)
    eTime = time.time()

    runtime = eTime - sTime
    print(f"======= Runtime Information =======")
    print(f"Runtime: {runtime} seconds")
    print(f"===================================")

# try:
#     customFileMain('yotel.csv')
# except:
#     traceback.print_exc() 