from flask import Flask, render_template, request, session, redirect, flash
from flask_bootstrap import Bootstrap
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
# import statsmodels.api as sm
import os
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from flask import json
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import contextily as ctx 
import geopandas as gpd 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from itertools import islice
import globalVar,ast
from startCustom import customFileMain

app = Flask(__name__)
Bootstrap(app)
app.secret_key = 'This is your secret key to utilize session in Flask'

# selectors for correlation ranking
SEASONS = ['winter', 'spring', 'autumn', 'summer']
GENERAL = ['average.rating', 'average.reviews.length', 'prices', 'amenities', 'province', 'holiday']

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
    return sentiment_piechart,int(positiveSent),int(negativeSent),int(totalSent),int(neutralSent)

def accomodationPieChart(csvFile):
    df = pd.read_csv(csvFile)
    # Assuming the CSV has a 'Category' column, get the top 4 values
    top_categories = df[globalVar.CATEGORIES].value_counts().nlargest(4)
    # Create the Plotly pie chart
    pie_chart = go.Figure(data=[go.Pie(labels=top_categories.index, values=top_categories)])
    # Convert the chart to an HTML div
    pie_chart_div = pie_chart.to_html(full_html=False)

    return pie_chart_div, top_categories.to_dict()

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

    wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    img_buffer = BytesIO()
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    wordcloud = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return wordcloud,word_freq

def amenitiesWordCloud(csvFile):
    df = pd.read_csv(csvFile)
    amenities_dict = {}
    amenities = df[globalVar.AMENITIES].tolist()
    for hotelamenities in amenities:
        amenity_list = ast.literal_eval(hotelamenities)
        for amenity in amenity_list:
            if amenity in amenities_dict:
                amenities_dict[amenity] += 1
            else:
                amenities_dict[amenity] = 1
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(amenities_dict)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    img_buffer = BytesIO()
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    wordcloud = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    return wordcloud, amenities_dict


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

def pricingHistogram(csvFile,dfHeader):
    df = pd.read_csv(csvFile)
    df = df[df[globalVar.PRICES] != 0]
    histogram_data = df[dfHeader].astype(float)
    histHeader = "Average Pricing" if dfHeader == globalVar.PRICES else "Pricing"

    # Create a histogram figure using go.Figure
    histogram = go.Figure(data=[go.Histogram(x=histogram_data)])
    histogram.update_layout(
        title=f'Bargraph of {histHeader}',
        xaxis_title= f"{histHeader}",
        yaxis_title='Hotels'
    )
    pricing_histogram = histogram.to_html()
    return pricing_histogram,histogram_data.mean()

def scatterplot():
    scattermap = px.scatter(df, x=globalVar.AVERAGE_RATING, y=globalVar.COMPOUND_SENTIMENT_SCORE, trendline='ols')
    return scattermap.to_html()

def provinceHistogram():
    count_province = df.groupby([globalVar.PROVINCE]).size().reset_index(name='Number of Hotels')
    max_province = count_province[globalVar.PROVINCE].max()
    max_province_count = count_province['Number of Hotels'].max()
    min_province = count_province[globalVar.PROVINCE].min()
    min_province_count = count_province['Number of Hotels'].min()
    provinces = px.bar(count_province, x='province', y='Number of Hotels')
    provinces.update_xaxes(tickangle=65)
    return provinces.to_html(), max_province, max_province_count, min_province, min_province_count

def createMap():
    grouped_df = df.groupby(globalVar.PROVINCE)[globalVar.REVIEWS_TOTAL].sum().reset_index()
    max_province = grouped_df[globalVar.PROVINCE].max()
    max_review_val = grouped_df[globalVar.REVIEWS_TOTAL].max()
    min_province = grouped_df[globalVar.PROVINCE].min()
    min_review_val = grouped_df[globalVar.REVIEWS_TOTAL].min()
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
    return map_div, max_review_val, max_province, min_review_val, min_province

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
        insight = (f"{hotelname} performing better than compared to the average hotel in general.")
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
        result = f"The word cloud for {hotelname} does not include common topics discussed in customer reviews."
        insight = (f"{hotelname} may have unique features or aspects that are not commonly discussed in reviews.\n " +
                f"Improvement can be considered in these areas: {all_hotel_keywords[:10]}")
    return background,result,insight

def getAmenitiesInsight(hotelname:str,hotel_amenities:dict,all_amenities:dict):
    background = ("A amenity word cloud shows what amenities are the most common among other hotels.")
    result = ""
    insight = ""
    all_amenities_sorted = dict(sorted(all_amenities.items(), key=lambda item: item[1], reverse=True))
    top10_amenities = list(all_amenities_sorted.keys())[:10]
    hotel_amenity = list(hotel_amenities.keys())[:10]
    # Compare the hotel's amenities with top 10 most sought after amenities
    implemented_amenities = set(hotel_amenity).intersection(top10_amenities)
    unimplemented_amenities = [amenity for amenity in hotel_amenity if amenity not in top10_amenities]

    
    if implemented_amenities:
        result = f"The word cloud for {hotelname} includes common amenities implemented by other hotels.\nThese amenities include: "
        result += ", ".join(implemented_amenities)
        insight = (f"{hotelname} shares common amenities implemented by other hotels, which is a positive sign.\nHowever these amenities " + 
                    "can be considered: ")
        insight += ", ".join(unimplemented_amenities)
    else:
        result = f"The word cloud for {hotelname} does not include common amenities implemented by other hotels, which implies there is much room for improvement."
        insight = f"{hotelname} may have to consider implementing these amenities to compete against competitor hotels: "
        insight += ", ".join(unimplemented_amenities)
    return background,result,insight

def getPricingInsight(hotelname:str,specific:float,all:float):
    background = ("The pricing histogram shows the average prices for a room of a hotel")
    result = ""
    insight = ""
    # Hotel higher rprices than average
    if specific > all:
        result = f"{hotelname} has higher average prices compared to the average hotel. Several factors may justify its pricing:"
        result += "\n1. Location and Demand: Prime locations and high demand lead to higher prices."
        result += "\n2. Quality and Luxury: Premium services and amenities justify higher costs."
        result += "\n3. Room Types: Suites and special views can command higher rates."

        insight = f"If {hotelname} is outperforming in pricing but underperforming in other areas, consider the following improvements:"
        insight += "\n1. Enhance customer services and overall guest experience."
        insight += "\n2. Implement new amenities or renovate existing ones to add value."
        insight += "\n3. Reevaluate pricing to ensure it aligns with current services and amenities offered."
    # Hotel lower review rating than average
    else:
        result = f"{hotelname} has lower average prices compared to the average hotel. Several factors may explain its competitive pricing:"
        result += "\n1. Location and Demand: Lower-demand areas may have more competitive prices."
        result += "\n2. Cost-Efficiency: Effective cost management can result in lower pricing."
        result += "\n3. Simplified Amenities: Fewer amenities can lead to more affordable rates."

        insight = f"If {hotelname} is excelling in pricing but underperforming in other areas, consider the following enhancements:"
        insight += "\n1. Marketing and Promotion: Promote the hotel's affordability to attract more guests."
        insight += "\n2. Expand Amenities: Consider adding or enhancing amenities to improve the guest experience."
        insight += "\n3. Monitor Customer Feedback: Stay attuned to guest reviews to identify areas for improvement."
    return background,result,insight

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
    max_df = df[df[globalVar.COMPOUND_SENTIMENT_SCORE] == df[globalVar.COMPOUND_SENTIMENT_SCORE].max()]
    min_df = df[df[globalVar.COMPOUND_SENTIMENT_SCORE] == df[globalVar.COMPOUND_SENTIMENT_SCORE].min()]

    max_score = max_df[globalVar.COMPOUND_SENTIMENT_SCORE].tolist()
    min_score = min_df[globalVar.COMPOUND_SENTIMENT_SCORE].tolist()
    max_date = max_df[globalVar.REVIEWS_DATE].tolist()
    min_date = min_df[globalVar.REVIEWS_DATE].tolist()
    max_text = max_df[globalVar.REVIEWS_TEXT].tolist()
    min_text = min_df[globalVar.REVIEWS_TEXT].tolist()

    df = df.groupby(globalVar.REVIEWS_DATE)[globalVar.COMPOUND_SENTIMENT_SCORE].mean()\
        .reset_index(name=globalVar.COMPOUND_SENTIMENT_SCORE)  
    fig = go.Figure([go.Scatter(x=df[globalVar.REVIEWS_DATE], y=df[globalVar.COMPOUND_SENTIMENT_SCORE])])
    fig.update_layout(yaxis_range=[-1,1])
    #fig = px.histogram(df, x=df[globalVar.REVIEWS_DATE], y=df[globalVar.COMPOUND_SENTIMENT_SCORE], histfunc='avg')
    return fig.to_html(), max_score[0], min_score[0], max_date[0], min_date[0], max_text[0], min_text[0]

def getSentimentRatingOverall(p,n,neu,t):
    dict_vals = {'positive':p,'negative':n,'neutral':neu}
    dict_vals = dict(sorted(dict_vals.items(), reverse=True, key=lambda x:x[1]))
    print(dict_vals)
    keys = list(dict_vals.keys())
    items = list(dict_vals.values())

    sentiment = ''
    if keys[0] == 'neutral':
        sentiment = 'neutral'
    elif keys[0] == 'positive':
        sentiment = 'positive'
    else:
        sentiment = 'negative'
    
    # get percentage of domination
    biggest = round(items[0]/t * 100, 4)
    second =  round(items[1]/t * 100, 4)
    last = round(items[2]/t * 100, 4)

    fringe = ''
    # check for fringe cases, i.e. biggest == second, biggest==second==last
    if biggest == second:
        fringe = 'There is a split of sentiment. There is a split between ' + keys[0] + " and " + keys[1]
    
    if biggest == 33.33 and second == 33.33 and last == 33.33:
        fringe = 'There is a close equilibrium. All sentimental values are weighted closely.'
    
    domination_val = int(biggest-(second+last))
    domination = ''

    if domination_val in range(-40,-20):
        domination = 'very slightly'
    elif domination_val in range(-20,0):
        domination = 'slightly'
    elif domination_val in range(1,10):
        domination = 'moderately'
    elif domination_val in range(11,59):
        domination = 'very'
    elif domination_val in range(60,101):
        domination = 'overwhelmingly'

    domination += " " + sentiment

    return sentiment, domination, biggest, fringe

@app.route('/home', methods=("POST", "GET"))
def homePage():
    hotel_df = pd.read_csv(session['analyzed_hotels'], index_col=0)
    # hotel_df = pd.read_csv('../csvs/bluefish_analyzedhotels.csv')
    # display values
    hotel_name = hotel_df[globalVar.NAME].iloc[0]
    hotel_category = hotel_df[globalVar.CATEGORIES].iloc[0]
    hotel_total_reviews = hotel_df[globalVar.REVIEWS_TOTAL].iloc[0]
    hotel_average = hotel_df[globalVar.AVERAGE_RATING].iloc[0]
    hotel_compound = '{:.2f}'.format(float(hotel_df[globalVar.COMPOUND_SENTIMENT_SCORE].iloc[0]))
    hotel_compound = float(hotel_compound)
    # compute hotel overall sentiment
    hotel_overall = ''
    if hotel_compound > 0:
        hotel_overall = 'Positive. Visitors are inclined to have positive sentiment towards the hotel.'
    elif hotel_overall < 0:
        hotel_overall = "Negative. Visitors are inclined to have negative sentiment towards the hotel."
    elif hotel_compound == 0:
        hotel_overall = 'Neutral. Visitors generally have neither polarizing positive nor negative sentiment towards the hotel.'

    pcHeader = "Pie Chart" 
    rrHeader = "Review Rating"
    wcHeader = "Word Cloud"
    amHeader = "Amenities"
    asHeader = "Sentiment Over Time"
    accomo_piechart = accomodationPieChart(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
    specific_sentiment_piechart,positiveSent,negativeSent,totalSent,neutralSent = sentimentPieChart(session['analyzed_reviews'])
    # compare num of sentiment values to get to the most dominant
    sentiment, domination, biggest_sentiment, fringe = getSentimentRatingOverall(positiveSent,negativeSent,neutralSent,totalSent)
    special_case = ''
    if fringe:
        special_case = fringe

    specific_keywords_wordcloud,specific_wordcloud = keywordsWordCloud(session['analyzed_hotels'])
    # get len of keywords for dynamism
    len_occur = len(specific_wordcloud)
    specific_averagerating_histogram,specific_averageRating = averageRatingHistogram(session['analyzed_reviews'],globalVar.REVIEWS_RATING)
    specific_amenities_wordcloud,specific_amenities = amenitiesWordCloud(session['analyzed_hotels'])
    average_sentiment_over_time_graph, max_score, min_score, max_date, min_date, max_text, min_text = averageSentimentOverTime(session['analyzed_reviews'])

    # check if amenities in the first place
    if len(specific_amenities) < 0:
        amenities_paragraph = ''
    else:
        specific_averageRating = "{:.2f}".format(specific_averageRating)
        specific_amenities = list(specific_amenities.keys())
        list_of_amenites_string = ', '.join(specific_amenities)

        amenities_paragraph = "Similar to the word cloud above, but counts the most reoccuring amenities mentioned in the hotel\'s writen reviews.\n" 
        amenities_paragraph += "The amnenities listed here are amenities that have been noted by reviewers and should be considered as a important characteristic of the hotel.\n"
        amenities_paragraph += f"{list_of_amenites_string}\n"

        if sentiment=='positive':
            amenities_paragraph +='As the sentiment of the hotel is positive, these amenities of note could be considered as selling point of the hotel, or ones that considerably aid its positive score.'
        elif sentiment=='negative':
            amenities_paragraph +='As the sentiment of the hotel is negative, the hotel should look into these amenities as it could be discussed negatively in reviews as it it brought up frequently.'
        
    

    return render_template("home.html",
                           hotel_name=hotel_name,
                           hotel_category=hotel_category,
                           hotel_total_reviews=hotel_total_reviews,
                           hotel_average=hotel_average,
                           hotel_compound=hotel_compound,
                           hotel_overall=hotel_overall,
                           pcHeader=pcHeader,
                           rrHeader=rrHeader, 
                           wcHeader=wcHeader,
                           amHeader=amHeader,
                           asHeader=asHeader,
                           accomo_piechart = accomo_piechart,
                           specific_sentiment_piechart = specific_sentiment_piechart,
                           sentiment=sentiment,
                           domination=domination,
                           biggest_sentiment=biggest_sentiment,
                           special_case=special_case,
                           positiveSent=positiveSent,
                           negativeSent=negativeSent,
                           totalSent=totalSent,
                           neutralSent=neutralSent,
                           specific_keywords_wordcloud = specific_keywords_wordcloud,
                           specific_wordcloud=specific_wordcloud,
                           len_occur=len_occur,
                           specific_averagerating_histogram = specific_averagerating_histogram,
                           specific_averageRating=specific_averageRating,
                           specific_amenities_wordcloud=specific_amenities_wordcloud,
                           amenities_paragraph=amenities_paragraph,
                        #    main_hotel_details=main_hotel_details,
                           average_sentiment_over_time_graph = average_sentiment_over_time_graph,
                           max_score=max_score, 
                           min_score=min_score, 
                           ax_date=max_date,
                           max_date=max_date, 
                           min_date=min_date, 
                           max_text=max_text, 
                           min_text=min_text)
    

@app.route('/comparison', methods=("POST", "GET"))
def comparisonPage():
    hotel_df = pd.read_csv(session['analyzed_hotels'], index_col=0)
    hotelname = hotel_df[globalVar.NAME].to_string(index=False)
    map_div = createMap()
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
    prComparisonHeader = "Pricing Comparison"

    # Comparisons Charts
    all_sentiment_piechart,all_positiveSent,all_negativeSent,all_totalSent,all_neutral = sentimentPieChart(globalVar.ANALYSISREVIEWOUTPUTFULLFILE)
    specific_sentiment_piechart,specific_positiveSent,specific_negativeSent,specific_totalSent,specific_neutral = sentimentPieChart(session['analyzed_reviews'])
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

    all_amenities_wordcloud,all_amenities = amenitiesWordCloud(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
    specific_amenities_wordcloud,specific_amenities = amenitiesWordCloud(session['analyzed_hotels'])
    awcbg,awcresult,awcinsight=getAmenitiesInsight(hotelname,specific_amenities,all_amenities)
    awcbg = awcbg.replace('\n', '<br>')
    awcresult = awcresult.replace('\n', '<br>')
    awcinsight = awcinsight.replace('\n', '<br>')

    all_pricing_histogram,all_pricing = pricingHistogram(globalVar.ANALYSISHOTELOUTPUTFULLFILE,globalVar.PRICES)
    specific_pricing_histogram,specific_pricing = pricingHistogram(session['analyzed_hotels'],globalVar.PRICES)
    prbg,prresult,rrinsight=getPricingInsight(hotelname,specific_pricing,all_pricing)
    prbg = prbg.replace('\n', '<br>')
    prresult = prresult.replace('\n', '<br>')
    rrinsight = rrinsight.replace('\n', '<br>')

    


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
                           awcbg=awcbg,
                           awcresult=awcresult,
                           awcinsight=awcinsight,
                           wcbg=wcbg,
                           wcresult=wcresult,
                           wcinsight=wcinsight,
                           accomo_piechart = accomo_piechart,
                           hotelname=hotelname, 
                           scattermap=scattermap, 
                           provinces=provinces,
                           prbg=prbg,
                           prresult=prresult,
                           rrinsight=rrinsight,
                           all_pricing_histogram=all_pricing_histogram,
                           specific_pricing_histogram=specific_pricing_histogram,
                           all_amenities_wordcloud = all_amenities_wordcloud,
                           specific_amenities_wordcloud = specific_amenities_wordcloud,
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
                           prComparisonHeader = prComparisonHeader,
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
    return wordcloud, word_freq

def scoringHeatmap(csvFile, correlationFile):
    # get the provinces in the given analyzed hotels into a list
    df_hotel = pd.read_csv(csvFile)
    
    # get the values needed correlate the coefficient values against the avg sentiment and ratings
    count_avg_sentiment_score = df_hotel.groupby(globalVar.PROVINCE)[globalVar.COMPOUND_SENTIMENT_SCORE].mean().reset_index()
    count_avg_rating_score = df_hotel.groupby(globalVar.PROVINCE)[globalVar.AVERAGE_RATING].mean().reset_index()
    count_avg_reviews_per_hotel = df_hotel.groupby(globalVar.PROVINCE)[globalVar.REVIEWS_TOTAL].mean().reset_index()
    # get the provinces
    province_count = count_avg_sentiment_score[globalVar.PROVINCE].values
    
    # read correlation file, get all the provinces score along with the provinces
    df_correlation = pd.read_csv(correlationFile)
    province_correlation_score = df_correlation[df_correlation[globalVar.CORRVARIABLE].isin(province_count)]
    
    # sort all count and score by province name
    province_correlation_score = province_correlation_score.sort_values(globalVar.CORRVARIABLE)
    count_avg_sentiment_score = count_avg_sentiment_score.sort_values(globalVar.PROVINCE)
    count_avg_rating_score = count_avg_rating_score.sort_values(globalVar.PROVINCE)
    count_avg_reviews_per_hotel = count_avg_reviews_per_hotel.sort_values(globalVar.PROVINCE)
    
    # list the values
    provinces = province_correlation_score[globalVar.CORRVARIABLE].tolist()
    coefficient_score = province_correlation_score[globalVar.CORRCOEFFICIENT].tolist()
    avg_sentiment_score = count_avg_sentiment_score[globalVar.COMPOUND_SENTIMENT_SCORE].tolist()
    avg_rating_score = count_avg_rating_score[globalVar.AVERAGE_RATING].tolist()
    avg_reviews_per_hotel = count_avg_reviews_per_hotel[globalVar.REVIEWS_TOTAL].tolist()

    # add them into final dataframe for correlation
    final_df = pd.DataFrame(provinces, columns=['provinces'])
    final_df = final_df.assign(coefficient=coefficient_score)
    final_df = final_df.assign(sentiment=avg_sentiment_score)
    final_df = final_df.assign(rating=avg_rating_score)
    final_df = final_df.assign(reviews=avg_reviews_per_hotel)
    final_df = final_df.drop('provinces', axis=1)
    
    value_len = len(final_df.columns.tolist())
    # correlate the scores of all the coeffiecient
    hmap = final_df.corr()

    stack = hmap.unstack()
    stack_ordered = stack.sort_values(kind="quicksort", ascending=False)
    
    top_vals = stack_ordered.head(value_len + 1)
    cor_vals = top_vals.iloc[value_len]
    cor_var = ''
    for x in top_vals.index:
        cor_var = x
    # print(cor_vals)
    # print(cor_var)
    # plot the heatmap
    plt.figure(figsize=(10,10))
    column_head = ['Provinces Coefficient', 'Sentiment Score', 'Average Rating', 'Reviews']
    correlation_map = sns.heatmap(hmap, annot=True, xticklabels=column_head, yticklabels=column_head, cmap="coolwarm", fmt=".2f")
    correlation_map.set_title('Average Score Correlelation')
    correlation_map.xaxis.tick_top()

    # encode to html and png format
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    heatmap = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    return heatmap, cor_var, cor_vals

def rankingAmenities(csvFile, correlationFile):
    # read files
    df = pd.read_csv(csvFile)
    df_cor =  pd.read_csv(correlationFile)
    # create a list to remove provinces, holidays, general scoring coefficients
    # get the existing provinces within the current analyzed dataset
    provinces = df[globalVar.PROVINCE].unique()
    amenities_df = df_cor[~df_cor[globalVar.CORRVARIABLE].isin(provinces)] 
    amenities_df = amenities_df[~amenities_df[globalVar.CORRVARIABLE].isin(SEASONS)]
    amenities_df = amenities_df[~amenities_df[globalVar.CORRVARIABLE].isin(GENERAL)]

    # sort amenities by coefficient score
    amenities_df = amenities_df.sort_values(by=[globalVar.CORRCOEFFICIENT], ascending=False)
    
    # get top rated and bottom rated amenities 
    top_rated_amenities = amenities_df[globalVar.CORRVARIABLE].head(5)
    worst_rated_amenities = amenities_df[globalVar.CORRVARIABLE].tail(5)

    # plot graph
    rank_graph = px.bar(amenities_df, x=globalVar.CORRVARIABLE, y=globalVar.CORRCOEFFICIENT, title='Amenities Ranking')
    rank_graph.update_xaxes(tickangle=65)

    return rank_graph.to_html(), top_rated_amenities.values.tolist(), worst_rated_amenities.values.tolist()

def rankingGeneral(correlationFile, selector):
    # read correlation file
    df =  pd.read_csv(correlationFile)
    # get this variables for ranking

    # retrieve from correlation file and sort
    general_coeff_scores = df[df[globalVar.CORRVARIABLE].isin(selector)]
    general_coeff_scores = general_coeff_scores.sort_values(by=[globalVar.CORRCOEFFICIENT], ascending=False)

    # get the best and worst factor sentiment analysis
    best_factor = general_coeff_scores[globalVar.CORRVARIABLE].head(1)
    worst_factor= general_coeff_scores[globalVar.CORRVARIABLE].tail(1)
    
    # plot graph
    rank_graph = px.bar(general_coeff_scores, x=globalVar.CORRVARIABLE, y=globalVar.CORRCOEFFICIENT, title='Factor Performance')
    rank_graph.update_xaxes(tickangle=65)

    return rank_graph.to_html(), best_factor.tolist(), worst_factor.tolist()

@app.route('/general', methods=("POST", "GET"))
def generalPage():
    # check if theres a POST request from dropdown list
    hotel_name = df[globalVar.NAME].unique()
    all_keywords_wordcloud = ''
    len_occur = 10
    if request.method == 'POST':
        selected_hotel = request.form['hotelName-dropdown']
        
        if selected_hotel:
            all_keywords_wordcloud,all_wordcloud = keywordsWordCloudSpecific(globalVar.ANALYSISHOTELOUTPUTFULLFILE, selected_hotel)
            all_wordcloud = dict(islice(all_wordcloud.items(), 0,len_occur))
        else:
            all_keywords_wordcloud,all_wordcloud = keywordsWordCloud(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
            all_wordcloud = dict(islice(all_wordcloud.items(), 0,len_occur))
    else:
        selected_hotel = ''
        all_keywords_wordcloud,all_wordcloud = keywordsWordCloud(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
        all_wordcloud = dict(islice(all_wordcloud.items(), 0,len_occur))
        
    map_div, max_reviews, max_province, min_reviews, min_province = createMap()
    scattermap = scatterplot()
    provinces, max_province_histogram, max_count, min_province_histogram, min_count = provinceHistogram()
    accomo_piechart, top_categories = accomodationPieChart(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
    score_heatmap, top_cor, top_cor_val = scoringHeatmap(globalVar.ANALYSISHOTELOUTPUTFULLFILE, globalVar.CORRFULLFILE)
    
    mpHeader = "Map of Hotels in USA against the number of ratings"
    acHeader = "Categories of the accomodations"
    pcHeader = "Pie Chart" 
    rrHeader = "Review Rating"
    wcHeader = "Word Cloud"
    pvHeader = "Number of hotels per Provinces in USA<"
    amHeader = "Amenities"
    smHeader = "Scoring Heatmap"
    aaHeader = "Average Rating of the accomodations"
    raHeader = "Amenities Rank"
    scHeader = "Plot of Compound Sentiment against Average Rating"
    gsHeader = "General Factors Coefficient Score"
    ssHeader = "Seasonal Ranking"

    all_sentiment_piechart,all_positiveSent,all_negativeSent,all_totalSent,all_neutral = sentimentPieChart(globalVar.ANALYSISREVIEWOUTPUTFULLFILE)
    all_averagerating_histogram,all_averageRating = averageRatingHistogram(globalVar.ANALYSISHOTELOUTPUTFULLFILE, globalVar.AVERAGE_RATING)
    all_amenities_rank, best_amenities, worst_amenities = rankingAmenities(globalVar.ANALYSISREVIEWOUTPUTFULLFILE, globalVar.CORRFULLFILE)
    all_gs_rank, best_factor, worst_factor = rankingGeneral(globalVar.CORRFULLFILE, GENERAL)
    all_season_rank, best_season, worst_season = rankingGeneral(globalVar.CORRFULLFILE, SEASONS)

    map_desc = "This is a map to indicate the number of reviews occurring in any given states of the USA. The brighter the value, the greater the amount of reviews occuring.\n"
    map_desc += f"The province with the greatest amount of visitors is {max_province} with {max_reviews}.\n"
    map_desc += f"The province with the greatest amount of visitors is {min_province} with {min_reviews}.\n"
    map_desc += f"Hotels should be aware that a large majority of reviews come from {max_province} visitors. It may indicate a well performing ground for hotel businesses."
    map_desc += f" On the other hand, hotels owners should note that {min_province} has not much reviews, indicating that they may be a lack of businesses for hotels, or lack of hotels."

    top_categories_string = ""
    top_categories_string = ', '.join(top_categories)
    top_categories_count = list(map(str, top_categories.values()))
    top_categories_string += "They appeared:"
    top_categories_string += ', '.join(top_categories_count)
    top_categories_string += ' times respectively.'
    accom_desc = "This is a piechart indicating which categories are most common among each hotel. They are listed here:"
    accom_desc += top_categories_string
    accom_desc += "Hotel owners should keep track as to which type of accomodation is most popular whith reviewers."

    return render_template("general.html",
                           hotel_name=hotel_name,
                           selected_hotel=selected_hotel,
                           mpHeader=mpHeader,
                           acHeader=acHeader,
                           pcHeader=pcHeader,
                           rrHeader=rrHeader, 
                           wcHeader=wcHeader,
                           pvHeader=pvHeader,
                           amHeader=amHeader,
                           smHeader=smHeader,
                           aaHeader=aaHeader,
                           raHeader=raHeader,
                           scHeader=scHeader,
                           gsHeader=gsHeader,
                           ssHeader=ssHeader,                          
                           accomo_piechart = accomo_piechart,
                           accom_desc=accom_desc,
                           scattermap=scattermap, 
                           provinces=provinces,
                           max_province_histogram=max_province_histogram,
                           max_count=max_count,
                           min_province_histogram=min_province_histogram,
                           min_count=min_count,
                           all_sentiment_piechart = all_sentiment_piechart,
                           len_occur=len_occur,
                           all_keywords_wordcloud = all_keywords_wordcloud,
                           all_wordcloud = all_wordcloud,
                           all_averagerating_histogram = all_averagerating_histogram,
                           all_averageRating=all_averageRating,
                           all_amenities_rank = all_amenities_rank,
                           all_gs_rank = all_gs_rank,
                           all_season_rank = all_season_rank,
                           score_heatmap=score_heatmap,
                           top_cor=top_cor,
                           top_cor_val=top_cor_val,
                           map_div=map_div,
                           map_desc=map_desc,
                           best_amenities=best_amenities,
                           worst_amenities=worst_amenities,
                           best_factor=best_factor,
                           worst_factor=worst_factor,
                           best_season=best_season,
                           worst_season=worst_season)

@app.route('/summary', methods=("POST", "GET"))
def summaryPage():
    return render_template("summary.html")

if __name__ == "__main__":    
  app.run(debug=True)