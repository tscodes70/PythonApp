from flask import Flask, render_template, request, session, redirect, flash
from flask_bootstrap import Bootstrap
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import os
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from flask import json
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import contextily as ctx 
import geopandas as gpd 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import globalVar,ast
from startCustom import customFileMain

app = Flask(__name__)
Bootstrap(app)
app.secret_key = 'This is your secret key to utilize session in Flask'

# takes data from analysedhotel-DATE.csv and loads it into a pandas dataframe
fileread = os.path.join(globalVar.CSVD,globalVar.ANALYSISHOTELOUTPUTFULLFILE)
df = pd.read_csv(fileread, index_col=0)

# Load a geospatial dataset of the USA
tlfolder = "tl_2022_us_state"
TLFILEFULL = os.path.join(globalVar.CWD,rf'{tlfolder}\tl_2022_us_state.shp')
gdf = gpd.read_file(TLFILEFULL)

def get_hotel_details(s):
    filter = df[df['name'] == s]
    return (filter.values).tolist()

def averageSentimentOverTime(csvFile):
    df = pd.read_csv(csvFile)
    df = df.sort_values(by=globalVar.REVIEWS_DATE)
    df = df.groupby(globalVar.REVIEWS_DATE)[globalVar.COMPOUND_SENTIMENT_SCORE].mean()\
        .reset_index(name=globalVar.COMPOUND_SENTIMENT_SCORE)  
    fig = go.Figure([go.Scatter(x=df[globalVar.REVIEWS_DATE], y=df[globalVar.COMPOUND_SENTIMENT_SCORE])])
    fig.update_layout(yaxis_range=[-1,1])
    return fig.to_html()

# Charts
def sentimentPieChart(csvFile):
    df = pd.read_csv(csvFile)
    df[globalVar.COMPOUND_SENTIMENT_SCORE] = pd.to_numeric(df[globalVar.COMPOUND_SENTIMENT_SCORE])

    positiveSent = (df[globalVar.COMPOUND_SENTIMENT_SCORE] > 0).sum()
    negativeSent = (df[globalVar.COMPOUND_SENTIMENT_SCORE] < 0).sum()
    neutralSent = (df[globalVar.COMPOUND_SENTIMENT_SCORE] == 0).sum()
    totalSent = positiveSent + negativeSent + neutralSent
    pcLabels = ["Positive Sentiment", "Negative Sentiment", "Neutral Sentiment"]
    valList = [positiveSent,negativeSent,neutralSent]

    # Create the Plotly pie chart
    spc = go.Figure(data=[go.Pie(labels=pcLabels, values=valList)])
    # Convert the chart to an HTML div
    sentiment_piechart = spc.to_html(full_html=False)
    return sentiment_piechart,int(positiveSent),int(negativeSent),int(totalSent)

def accomodationPieChart(csvFile):
    df = pd.read_csv(csvFile)
    # Assuming the CSV has a 'Category' column, get the top 4 values
    top_categories = df[globalVar.CATEGORIES].value_counts().nlargest(4)
    # Create the Plotly pie chart
    pie_chart = go.Figure(data=[go.Pie(labels=top_categories.index, values=top_categories)])
    # Convert the chart to an HTML div
    pie_chart_div = pie_chart.to_html(full_html=False)

    return pie_chart_div

def keywordsWordCloud(csvFile):
    df = pd.read_csv(csvFile)
    keywords = df[globalVar.POPULAR_KEYWORDS].tolist()
    word_freq = {}
    for item in keywords:
        word_list = ast.literal_eval(item)  # Convert the string to a list of tuples
        for word, freq in word_list:
            if word in word_freq:
                word_freq[word] += int(freq)  # Convert freq to int and add to existing count
            else:
                word_freq[word] = int(freq)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    img_buffer = BytesIO()
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    wordcloud = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return wordcloud,word_freq

def averageRatingHistogram(csvFile,dfHeader):
    df = pd.read_csv(csvFile)
    histogram_data = df[dfHeader].astype(float)
    histHeader = "Average Rating" if dfHeader == globalVar.AVERAGE_RATING else "Rating"

    # Create a histogram figure using go.Figure
    histogram = go.Figure(data=[go.Histogram(x=histogram_data)])
    histogram.update_layout(
        title=f'Histogram of {histHeader}',
        xaxis_title= f"{histHeader}",
        yaxis_title='Count'
    )
    rating_histogram = histogram.to_html()
    return rating_histogram,histogram_data.mean()

def scatterplot():
    scattermap = px.scatter(df, x=globalVar.AVERAGE_RATING, y=globalVar.COMPOUND_SENTIMENT_SCORE)
    return scattermap.to_html()

def provinceHistogram():
    count_province = df.groupby([globalVar.PROVINCE]).size().reset_index(name='Number of Hotels')
    provinces = px.bar(count_province, x='province', y='Number of Hotels')
    return provinces.to_html()

def map():
    grouped_df = df.groupby('province')['reviews.total'].sum().reset_index()
    # Merge the data based on 'Province' and calculate the total number of reviews
    merged_data = gdf.merge(grouped_df, left_on='STUSPS', right_on='province', how='left')

    # Normalize the data if needed
    # Here, we are assuming you have a column 'Total Reviews' in your CSV data
    # You may want to scale the data to fit the color scale properly

    # Create a chloropleth map using Plotly Express
    fig = px.choropleth(merged_data, 
                        geojson=gdf.geometry, 
                        locations=merged_data.index, 
                        color='reviews.total',
                        hover_name='STUSPS',
                        range_color=[0,500])


    fig.update_geos(
        visible=False,
        projection_scale=1,
        center={"lat": 37.0902, "lon": -95.7129},
        scope="usa"
    )
    # Convert the map to HTML
    map_div = fig.to_html(full_html=False)
    return map_div

def getSentimentInsight(hotelname:str,specificList:list,allList:list):
    background = ("<b>Positive Sentiment:</b> This category is used to describe customer reviews that express a favorable or optimistic attitude. It often indicates happiness, satisfaction, approval, or agreement.\n" +
                  "<b>Negative Sentiment:</b> Negative sentiment represents customer reviews that convey a critical, unfavorable, or pessimistic attitude. It is typically associated with dissatisfaction, disapproval, or disagreement.\n" +
                  "<b>Neutral Sentiment:</b> Neutral sentiment refers to customer reviews that do not express a strong positive or negative emotional tone. It often indicates a lack of strong emotion or a balanced viewpoint.")
    result = ""
    insight = ""
    # Hotel more positive sentiment than average
    if specificList[0]/specificList[2] > allList[0]/allList[2]:
        result = (f"{hotelname} has more satisfied customers than the average hotel. It is likely that " +
                  f"{hotelname} provides:\n" +
                  "1. High Quality of Services\n" +
                  "2. Competitve Pricing\n" + 
                  "3. Special Features or Amenities")
        insight = (f"{hotelname} performing better than the average hotel in terms of customer satisfaction, not much insight to be given for this comparison.")
    # Hotel lesser positive sentiment than average
    else:
        result = (f"{hotelname} has lesser satisfied customers than the average hotel. It is likely that " +
                  f"{hotelname} is lacking in:\n" +
                  "1. High Quality of Services\n" +
                  "2. Competitve Pricing\n" + 
                  "3. Special Features or Amenities")
        insight = (f"{hotelname} is underperforming compared to the average hotel in terms of customer satisfaction.\n" +
                   f"Possible variables to look into:\n " +
                    "1. Enhancing Customer Services \n" +
                    "2. Increasing engagement with Customers\n" +
                    "3. Maintaining Clean and Safe Enviroments\n" +
                    "4. Staff Motivation and Training\n" +
                    "5. Monitoring Online Reputation")
    return background,result,insight

def getReviewRatingInsight(hotelname:str,specific:float,all:float):
    background = ("The average rating of hotel reviews is a numerical representation of the overall satisfaction or evaluation of a hotel based on multiple reviews. It is calculated by summing up the ratings or scores given by individual reviewers and dividing the total by the number of reviews.")
    result = ""
    insight = ""
    # Hotel higher review rating than average
    if specific > all:
        result = (f"{hotelname} has a higher average rating than the average hotel. It is likely that " +
                  f"{hotelname} provides:\n" +
                  "1. Positive Guest Experience\n" +
                  "2. Quality Service\n" + 
                  "3. Good Value for Money")
        insight = (f"{hotelname} performing better than the average hotel in terms of customer satisfaction, not much insight to be given for this comparison.")
    # Hotel lower review rating than average
    else:
        result = (f"{hotelname} has a lower average rating than the average hotel. It is likely that " +
                  f"{hotelname} is lacking in:\n" +
                  "1. Positive Guest Experience\n" +
                  "2. Quality Service\n" + 
                  "3. Good Value for Money")
        insight = (f"{hotelname} is underperforming compared to the average hotel in general.\n" +
                   f"Possible variables to look into:\n " +
                    "1. Enhancing Customer Services \n" +
                    "2. Increasing engagement with Customers\n" +
                    "3. Maintaining Clean and Safe Enviroments\n" +
                    "4. Staff Motivation and Training\n" +
                    "5. Monitoring Online Reputation")
    return background,result,insight

def getWordCloudInsight(hotelname:str,hotel_keywords:list,all_hotel_keywords:list):
    background = ("A Wordcloud shows the most discussed topics of customers through their reviews")
    result = ""
    insight = ""
    # Compare the hotel's keywords with all hotel keywords
    common_keywords = set(hotel_keywords).intersection(all_hotel_keywords)
    
    if common_keywords:
        result = f"The word cloud for {hotelname} includes common topics discussed in customer reviews.\nThese topics may include: "
        result += ", ".join(common_keywords)
        insight = (f"{hotelname} shares common discussion topics with other hotels, which is a positive sign as" + 
                    " it is: \n" +
                    "1. Meeting Industry Standards\n" +
                    "2. Consistent with competitors")
    else:
        result = f"The word cloud for {hotelname} does not include common topics discussed in customer reviews. This may indicate unique aspects."
        insight = f"{hotelname} may have unique features or aspects that are not commonly discussed in reviews."
    return background,result,insight

@app.route('/filtered_charts', methods=['GET','POST'])
def filtered_charts():
    pie_chart_div = accomodationPieChart(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
    selected_hotel = request.form['hotelName-dropdown']

    # histogram
    #if selected_hotel:
    #    histogram_filtered_data = df[df['name'] == selected_hotel]
    #    histogram_data = histogram_filtered_data['average.rating'].astype(float)
    #else:
    histogram_data = df['average.rating'].astype(float)

    # Create a histogram figure using go.Figure
    histogram = go.Figure(data=[go.Histogram(x=histogram_data)])
    histogram.update_layout(
        title=f'Histogram of Average Rating',
        xaxis_title='Average Rating',
        yaxis_title='Count'
    )
    histogram_div = histogram.to_html()
    
    # wordcloud
    hotel_name = df['name'].unique()
    wordcloud_heading = "Word Cloud on the reviews for " + selected_hotel
    histogram_heading = "Histogram on the ratings for " + selected_hotel

    if selected_hotel:
        wordcloud_filtered_data = df[df['name'] == selected_hotel]
        wordcloud_data = ' '.join(wordcloud_filtered_data['reviews.summary'].astype(str))
    else:
        wordcloud_data = ' '.join(df['reviews.summary'].astype(str))

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400)
    wordcloud.generate(wordcloud_data)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    img_buffer = BytesIO()
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    hotel_details = []
    #get hotel values for display
    if selected_hotel:
        hotel_details = get_hotel_details(selected_hotel)

    # set selected_hotel to 'All' for presentation
    if not selected_hotel:
        selected_hotel = 'All hotels'
    
    scattermap = scatterplot()
    provinces = provinceHistogram()

    return render_template('userDashboard copy.html', img_data=img_data, hotelNames= hotel_name, 
                           histogram_heading=histogram_heading,wordcloud_heading=wordcloud_heading, 
                           histogram_div=histogram_div, hotelName=selected_hotel, pie_chart_div=pie_chart_div,
                           hotel_details=hotel_details, scattermap=scattermap, provinces=provinces)

@app.route('/', methods=['GET','POST'])
def upload():
    try:
        if request.method == 'POST':
            #check if POST request has file
            if 'file' not in request.files:
                flash('Invalid file upload')
                return redirect('upload.html')
            # get file from POST
            f = request.files.get('file')
            filename = secure_filename(f.filename)
            f.save(os.path.join(globalVar.CSVD, filename))

            # 1) user will upload their own related hotel csv
            session['hotel_file'] = os.path.join(globalVar.CSVD,filename)
            # 2) csv will get cleaned, analyzed and home page will read wtv yall need from examplehotelname_analyzedreviews_12-Oct.csv and examplehotelname_analyzedhotels_12-Oct.csv
            hoteldata = filename[:-4]
            # insert cleaning and analysis here
            customFileMain(filename)
            
            #Session stuff 
            session['analyzed_hotels'] = os.path.join(globalVar.CSVD,f"{hoteldata}_analyzedhotels.csv")
            session['analyzed_reviews'] = os.path.join(globalVar.CSVD,f"{hoteldata}_analyzedreviews.csv")

            return redirect('/home')
    except Exception:
        print("error")
    return render_template("upload.html")

@app.route('/api/general')
def summary():
     dfc = df.copy()
     dfc.columns = dfc.columns.str.replace('.', '_')
     jsonfile = dfc.to_json(orient='table')
     return jsonfile

def averageSentimentOverTime(csvFile):
    df = pd.read_csv(csvFile)
    df = df.sort_values(by=globalVar.REVIEWS_DATE)
    df = df.groupby(globalVar.REVIEWS_DATE)[globalVar.COMPOUND_SENTIMENT_SCORE].mean()\
        .reset_index(name=globalVar.COMPOUND_SENTIMENT_SCORE)  
    fig = go.Figure([go.Scatter(x=df[globalVar.REVIEWS_DATE], y=df[globalVar.COMPOUND_SENTIMENT_SCORE])])
    fig.update_layout(yaxis_range=[-1,1])
    #fig = px.histogram(df, x=df[globalVar.REVIEWS_DATE], y=df[globalVar.COMPOUND_SENTIMENT_SCORE], histfunc='avg')
    return fig.to_html()

@app.route('/home', methods=("POST", "GET"))
def homePage():
    hotel_df = pd.read_csv(session['analyzed_hotels'], index_col=0)
    main_hotel_details = hotel_df.values.tolist()
    pcHeader = "Pie Chart" 
    rrHeader = "Review Rating"
    wcHeader = "Word Cloud"
    amHeader = "Amenities"
    asHeader = "Sentiment Over Time"
    accomo_piechart = accomodationPieChart(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
    specific_sentiment_piechart,positiveSent,negativeSent,totalSent = sentimentPieChart(session['analyzed_reviews'])
    specific_keywords_wordcloud,specific_wordcloud = keywordsWordCloud(session['analyzed_hotels'])
    specific_averagerating_histogram,specific_averageRating = averageRatingHistogram(session['analyzed_reviews'],globalVar.REVIEWS_RATING)
    average_sentiment_over_time_graph = averageSentimentOverTime(session['analyzed_reviews'])

    return render_template("home.html",
                           pcHeader=pcHeader,
                           rrHeader=rrHeader, 
                           wcHeader=wcHeader,
                           amHeader=amHeader,
                           asHeader=asHeader,
                           accomo_piechart = accomo_piechart,
                           specific_sentiment_piechart = specific_sentiment_piechart,
                           specific_keywords_wordcloud = specific_keywords_wordcloud,
                           specific_averagerating_histogram = specific_averagerating_histogram,
                           main_hotel_details=main_hotel_details,
                           average_sentiment_over_time_graph = average_sentiment_over_time_graph)

@app.route('/comparison', methods=("POST", "GET"))
def comparisonPage():
    hotel_df = pd.read_csv(session['analyzed_hotels'], index_col=0)
    hotelname = hotel_df[globalVar.NAME].to_string(index=False)
    map_div = map()
    scattermap = scatterplot()
    provinces = provinceHistogram()
    accomo_piechart = accomodationPieChart(globalVar.ANALYSISHOTELOUTPUTFULLFILE)

    hotel_df = pd.read_csv(session['analyzed_hotels'], index_col=0)
    main_hotel_details = hotel_df.values.tolist()
    pcHeader = "Pie Chart" 
    rrHeader = "Review Rating"
    wcHeader = "Word Cloud"
    amHeader = "Amenities"

    # Comparisons
    pcComparisonHeader = "Pie Chart Comparison"
    wcComparisonHeader = "Word Cloud Comparison"
    rrComparisonHeader = "Review Rating Comparison"
    amComparisonHeader = "Amenities Comparison"

    # Comparisons Charts
    all_sentiment_piechart,all_positiveSent,all_negativeSent,all_totalSent = sentimentPieChart(globalVar.ANALYSISREVIEWOUTPUTFULLFILE)
    specific_sentiment_piechart,specific_positiveSent,specific_negativeSent,specific_totalSent = sentimentPieChart(session['analyzed_reviews'])
    sentbg,sentresult,sentinsight=getSentimentInsight(hotelname,[specific_positiveSent,specific_negativeSent,specific_totalSent],[all_positiveSent,all_negativeSent,all_totalSent])
    sentbg = sentbg.replace('\n', '<br>')
    sentresult = sentresult.replace('\n', '<br>')
    sentinsight = sentinsight.replace('\n', '<br>')

    all_keywords_wordcloud,all_wordcloud = keywordsWordCloud(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
    specific_keywords_wordcloud,specific_wordcloud = keywordsWordCloud(session['analyzed_hotels'])
    wcbg,wcresult,wcinsight=getWordCloudInsight(hotelname,specific_wordcloud,all_wordcloud)
    wcbg = wcbg.replace('\n', '<br>')
    wcresult = wcresult.replace('\n', '<br>')
    wcinsight = wcinsight.replace('\n', '<br>')

    all_averagerating_histogram,all_averageRating = averageRatingHistogram(globalVar.ANALYSISHOTELOUTPUTFULLFILE,globalVar.AVERAGE_RATING)
    specific_averagerating_histogram,specific_averageRating = averageRatingHistogram(session['analyzed_reviews'],globalVar.REVIEWS_RATING)

    arbg,arresult,arinsight=getReviewRatingInsight(hotelname,specific_averageRating,all_averageRating)
    arbg = arbg.replace('\n', '<br>')
    arresult = arresult.replace('\n', '<br>')
    arinsight = arinsight.replace('\n', '<br>')

    return render_template("comparison.html",
                           pcHeader=pcHeader,
                           rrHeader=rrHeader, 
                           wcHeader=wcHeader,
                           amHeader=amHeader,
                           sentbg = sentbg,
                           sentresult=sentresult,
                           sentinsight=sentinsight,
                           arbg=arbg,
                           arresult=arresult,
                           arinsight=arinsight,
                           wcbg=wcbg,
                           wcresult=wcresult,
                           wcinsight=wcinsight,
                           accomo_piechart = accomo_piechart,
                           hotelname=hotelname, 
                           scattermap=scattermap, 
                           provinces=provinces, 
                           all_sentiment_piechart = all_sentiment_piechart,
                           specific_sentiment_piechart = specific_sentiment_piechart,
                           all_keywords_wordcloud = all_keywords_wordcloud,
                           specific_keywords_wordcloud = specific_keywords_wordcloud,
                           all_averagerating_histogram = all_averagerating_histogram,
                           specific_averagerating_histogram = specific_averagerating_histogram,
                           pcComparisonHeader = pcComparisonHeader, 
                           rrComparisonHeader = rrComparisonHeader, 
                           wcComparisonHeader = wcComparisonHeader,
                           amComparisonHeader = amComparisonHeader,
                           main_hotel_details=main_hotel_details,
                           map_div=map_div)

def keywordsWordCloudSpecific(csvFile, selector):
    df = pd.read_csv(csvFile)
    wordcloud_filtered_data = df[df['name'] == selector].copy()
    # wordcloud_data = ' '.join(wordcloud_filtered_data[globalVar.POPULAR_KEYWORDS].astype(str))
    keywords = wordcloud_filtered_data[globalVar.POPULAR_KEYWORDS].tolist()
    # wordcloud = WordCloud(width=800, height=400, background_color='white')
    # wordcloud.generate(wordcloud_data)
    word_freq = {}
    for item in keywords:
         word_list = ast.literal_eval(item)  # Convert the string to a list of tuples
         for word, freq in word_list:
             if word in word_freq:
                 word_freq[word] += int(freq)  # Convert freq to int and add to existing count
             else:
                 word_freq[word] = int(freq)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    img_buffer = BytesIO()
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    wordcloud = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return wordcloud

@app.route('/general', methods=("POST", "GET"))
def generalPage():
    # check if theres a POST request from dropdown list
    hotel_name = df[globalVar.NAME].unique()
    all_keywords_wordcloud = ''
    if request.method == 'POST':
        selected_hotel = request.form['hotelName-dropdown']
        
        if selected_hotel:
            all_keywords_wordcloud = keywordsWordCloudSpecific(globalVar.ANALYSISHOTELOUTPUTFULLFILE, selected_hotel)
        else:
            all_keywords_wordcloud,all_wordcloud = keywordsWordCloud(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
    else:
        selected_hotel = ''
        all_keywords_wordcloud,all_wordcloud = keywordsWordCloud(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
        
    map_div = map()
    scattermap = scatterplot()
    provinces = provinceHistogram()
    accomo_piechart = accomodationPieChart(globalVar.ANALYSISHOTELOUTPUTFULLFILE)

    pcHeader = "Pie Chart" 
    rrHeader = "Review Rating"
    wcHeader = "Word Cloud"
    amHeader = "Amenities"

    all_sentiment_piechart = sentimentPieChart(globalVar.ANALYSISREVIEWOUTPUTFULLFILE)
    all_averagerating_histogram,all_averageRating = averageRatingHistogram(globalVar.ANALYSISHOTELOUTPUTFULLFILE,globalVar.AVERAGE_RATING)

    return render_template("general.html",
                           hotel_name=hotel_name,
                           selected_hotel=selected_hotel,
                           pcHeader=pcHeader,
                           rrHeader=rrHeader, 
                           wcHeader=wcHeader,
                           amHeader=amHeader,
                           accomo_piechart = accomo_piechart,
                           scattermap=scattermap, 
                           provinces=provinces, 
                           all_sentiment_piechart = all_sentiment_piechart,
                           all_keywords_wordcloud = all_keywords_wordcloud,
                           all_averagerating_histogram = all_averagerating_histogram,
                           map_div=map_div)

@app.route('/summary', methods=("POST", "GET"))
def summaryPage():
    return render_template("summary.html")

if __name__ == "__main__":    
  app.run(debug=True)