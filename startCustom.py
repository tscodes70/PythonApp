import dataClean,dataStandardization,dataAnalyze
import time,traceback,globalVar,os,datetime

def customFileMain():
    sTime = time.time()
    currentdate = datetime.date.today().strftime("%d-%b")
    hotelname = "yotel"

    hoteldata = "yotel_hotel_test.csv"
    hotelclean = f"{hotelname}_clean_{currentdate}.csv"
    hotelmd = f"{hotelname}_md_{currentdate}.csv"
    hotelar = f"{hotelname}_analyzedreviews_{currentdate}.csv"
    hotelah = f"{hotelname}_analyzedhotels_{currentdate}.csv"

        
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

try:
    customFileMain()

except:
    traceback.print_exc() 