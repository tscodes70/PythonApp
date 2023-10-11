from flask import Flask, render_template, request, session, redirect
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
import globalVar

app = Flask(__name__)
Bootstrap(app)

# takes data from outputdata.csv and loads it into a pandas dataframe
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

    return render_template('userDashboard copy.html', img_data=img_data, hotelNames= hotel_name, 
                           histogram_heading=histogram_heading,wordcloud_heading=wordcloud_heading, 
                           histogram_div=histogram_div, hotelName=selected_hotel, pie_chart_div=pie_chart_div,
                           hotel_details=hotel_details)


@app.route('/', methods=['GET', 'POST'])
def navigation():
    pie_chart_div = category_piechart()
    #map_div = map()
    histogram_heading = "Histogram"
    wordcloud_heading = "Word Cloud"

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
    text = ' '.join(df['reviews.summary'].astype(str))

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400)
    wordcloud.generate(text)

    # Render the word cloud as a base64-encoded image
    img_buffer = BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    
    # Perform rank correlation analysis (e.g., Spearman's rank correlation)
    # Replace 'column1' and 'column2' with the columns you want to analyze
    correlation_value = df[globalVar.COMPOUND_SENTIMENT_SCORE].corr(df[globalVar.AVERAGE_RATING], method='spearman')

    # Create a Plotly scatter plot with the correlation value
    correlationgraph = go.Figure(data=go.Scatter(x=[0], y=[correlation_value], mode='markers+text'))
    correlationgraph.update_layout(
        title='Rank Correlation',
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        showlegend=False
    )
    correlationgraph = correlationgraph.to_html()
    
    return render_template('userDashboard copy.html',  histogram_heading=histogram_heading, histogram_div=histogram_div, 
                           wordcloud_heading = wordcloud_heading, hotelNames=hotel_name, 
                           pie_chart_div=pie_chart_div, img_data=img_data)

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