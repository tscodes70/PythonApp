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

app = Flask(__name__)
Bootstrap(app)
app.secret_key = 'This is your secret key to utilize session in Flask'

# allow upload to csvs folder
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csvs/'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# takes data from analysedhotel-DATE.csv and loads it into a pandas dataframe
fileread = os.path.join(globalVar.CSVD,globalVar.ANALYSISHOTELOUTPUTFULLFILE)
df = pd.read_csv(fileread, index_col=0)

# Load a geospatial dataset of the USA
tlfolder = "tl_2022_us_state"
TLFILEFULL = os.path.join(globalVar.CWD,rf'{tlfolder}\tl_2022_us_state.shp')
gdf = gpd.read_file(TLFILEFULL)

def category_piechart():
    # Assuming the CSV has a 'Category' column, get the top 4 values
    top_categories = df['categories'].value_counts().nlargest(4)
    # Create the Plotly pie chart
    pie_chart = go.Figure(data=[go.Pie(labels=top_categories.index, values=top_categories)])
    # Convert the chart to an HTML div
    pie_chart_div = pie_chart.to_html(full_html=False)

    return pie_chart_div

def get_hotel_details(s):
    filter = df[df['name'] == s]
    return (filter.values).tolist()

def scatterplot():
    scattermap = px.scatter(df, x=globalVar.AVERAGE_RATING, y=globalVar.COMPOUND_SENTIMENT_SCORE)
    return scattermap.to_html()

def bargraph():
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

@app.route('/filtered_charts', methods=['GET','POST'])
def filtered_charts():
    pie_chart_div = category_piechart()
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
    provinces = bargraph()

    return render_template('userDashboard copy.html', img_data=img_data, hotelNames= hotel_name, 
                           histogram_heading=histogram_heading,wordcloud_heading=wordcloud_heading, 
                           histogram_div=histogram_div, hotelName=selected_hotel, pie_chart_div=pie_chart_div,
                           hotel_details=hotel_details, scattermap=scattermap, provinces=provinces)

# Charts
def sentimentPieChart(csvFile):
    df = pd.read_csv(csvFile)
    df[globalVar.COMPOUND_SENTIMENT_SCORE] = pd.to_numeric(df[globalVar.COMPOUND_SENTIMENT_SCORE])
    positiveSent = (df[globalVar.COMPOUND_SENTIMENT_SCORE] > 0).sum()
    negativeSent = (df[globalVar.COMPOUND_SENTIMENT_SCORE] < 0).sum()
    neutralSent = (df[globalVar.COMPOUND_SENTIMENT_SCORE] == 0).sum()
    pcLabels = ["Positive Sentiment", "Negative Sentiment", "Neutral Sentiment"]
    valList = [positiveSent,negativeSent,neutralSent]

    # Create the Plotly pie chart
    spc = go.Figure(data=[go.Pie(labels=pcLabels, values=valList)])
    # Convert the chart to an HTML div
    sentiment_piechart = spc.to_html(full_html=False)
    return sentiment_piechart

def wordCloud(csvFile):
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
    return wordcloud


@app.route('/dashboard', methods=['GET', 'POST'])
def navigation():
    # Home page will render info related to a singular hotel : hotel info, word map for single hotel, etc
    # General overview will return all info related to all hotels : selected hotel info, word map (can be selected), graphs, table

    # home
    # read file name from session
    hotel_df = pd.read_csv(session['uploaded_data_file_path'], index_col=0)
    # read this in main home 
    main_hotel_details = hotel_df.values.tolist()
    # make a word cloud specific to file session hotel
    t = ' '.join(hotel_df[globalVar.GREVIEWS_SUMMARY].astype(str))
    wordcloud = WordCloud(width=800, height=400)
    wordcloud.generate(t)
    img_buffer = BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    pie_chart_div = category_piechart()
    map_div = map()
    histogram_heading = "Histogram"
    wordcloud_heading = "Word Cloud"

    # Comparisons
    pcComparisonHeader = "Pie Chart Comparison"
    wcComparisonHeader = "Word Cloud Comparison"
    rrComparisonHeader = "Review Rating Comparison"
    amComparisonHeader = "Amenities Comparison"

    # Comparisons Charts
    all_sentiment_piechart = sentimentPieChart(globalVar.ANALYSISREVIEWOUTPUTFULLFILE)
    specific_sentiment_piechart = sentimentPieChart(session['analyzed_reviews'])

    all_wordcloud = wordCloud(globalVar.ANALYSISHOTELOUTPUTFULLFILE)
    specific_wordcloud = wordCloud(session['analyzed_hotels'])

    hotel_name = df['name'].unique()
    
    # Create a histogram figure using go.Figure
    histogram_data = df['average.rating'].astype(float)
    histogram = go.Figure(data=[go.Histogram(x=histogram_data)])
    histogram.update_layout(
        title=f'Histogram of Average Rating',
        xaxis_title='Average Rating',
        yaxis_title='Count'
    )
    histogram_div = histogram.to_html()
    
    # Generate the word cloud
    # text = ' '.join(df['reviews.summary'].astype(str))

    # Generate the word cloud
    # wordcloud = WordCloud(width=800, height=400)
    # wordcloud.generate(text)

    # Render the word cloud as a base64-encoded image
    # img_buffer = BytesIO()
    # plt.figure(figsize=(8, 4))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.savefig(img_buffer, format='png')
    # img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    scattermap = scatterplot()
    provinces = bargraph()

    return render_template('userDashboard copy.html',
                           histogram_heading=histogram_heading, 
                           histogram_div=histogram_div, 
                           wordcloud_heading = wordcloud_heading, 
                           hotelNames=hotel_name, 
                           pie_chart_div=pie_chart_div, 
                           scattermap=scattermap, 
                           img_data=img_data,
                           provinces=provinces, 

                           all_sentiment_piechart = all_sentiment_piechart,
                           specific_sentiment_piechart = specific_sentiment_piechart,
                           all_wordcloud = all_wordcloud,
                           specific_wordcloud = specific_wordcloud,
                           pcComparisonHeader = pcComparisonHeader, 
                           rrComparisonHeader = rrComparisonHeader, 
                           wcComparisonHeader = wcComparisonHeader,
                           amComparisonHeader = amComparisonHeader,
                           main_hotel_details=main_hotel_details,
                           map_div=map_div)

@app.route('/', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
	    #check if POST request has file
        if 'file' not in request.files:
            flash('Invalid file upload')
            return redirect('new_upload.html')
        # get file from POST
        f = request.files.get('file')
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #save the stuff and redirect to dashboard and save the file path to session for reading
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],
					filename)
        
        # 1) user will upload their own related hotel csv
        # 2) csv will get cleaned, analyzed and home page will read wtv yall need from examplehotelname_analyzedreviews_12-Oct.csv and examplehotelname_analyzedhotels_12-Oct.csv
        # insert cleaning and analysis here

        
        #Session stuff 
        session['analyzed_hotels'] = os.path.join(app.config['UPLOAD_FOLDER'],"yotel_analyzedhotels_12-Oct.csv")
        session['analyzed_reviews'] = os.path.join(app.config['UPLOAD_FOLDER'],"yotel_analyzedreviews_12-Oct.csv")
        return redirect('/dashboard')

    return render_template("new_upload.html")

@app.route('/api/general')
def summary():
     dfc = df.copy()
     dfc.columns = dfc.columns.str.replace('.', '_')
     jsonfile = dfc.to_json(orient='table')
     return jsonfile

@app.route('/viewSummary', methods=("POST", "GET"))
def index():
    return render_template("viewSummary.html", title="View Summary")
   
if __name__ == "__main__":    
  app.run(debug=True)