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
#import numpy as np

'''
this is a seperate page to render the user preferance table page;
this will be integrated onto the main app.py once all features are completed
done:
render table
to-do:
- add some predictive analysis???
- clicking onto the hotel loads a page which renders all its info and analysis
-- required:
-- link to a new page??
-- page renders info?
- use ajax query to load backend for faster loading without breaking jinjia
'''

app = Flask(__name__)
Bootstrap(app)


#Configure upload file path flask
UPLOAD_FOLDER = os.path.join('dataset')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# takes data from outputdata.csv and loads it into a pandas dataframe
df = pd.read_csv('analyzedhotels_10-Oct.csv', index_col=0)

# Load a geospatial dataset of the USA
gdf = gpd.read_file(r'C:\Users\legendary22\Documents\GitHub\PythonApp\tl_2022_us_state\tl_2022_us_state.shp')

# Route to get unique provinces for the dropdown
@app.route('/get_provinces')
def get_provinces():
    provinces = df['Province'].unique()
    return render_template('userDashboard.html', provinces=provinces)

# Route to filter data based on the selected province
@app.route('/filter_data')
def filter_data(): 
    selected_province = request.args.get('province')

    if selected_province:
        filtered_data = df[df['Province'] == selected_province]
    else:
        filtered_data = df

    filtered_data_html = filtered_data.to_html()
    return filtered_data_html

def piechart():
    # Assuming the CSV has a 'Category' column, get the top 4 values
    top_categories = df['categories'].value_counts().nlargest(4)
    # Create the Plotly pie chart
    pie_chart = go.Figure(data=[go.Pie(labels=top_categories.index, values=top_categories)])
    # Convert the chart to an HTML div
    pie_chart_div = pie_chart.to_html(full_html=False)

    return pie_chart_div

@app.route('/filtered_charts', methods=['POST'])
def filtered_charts():
    pie_chart_div = piechart()
    selected_hotel = request.form['hotelName-dropdown']
    
    # histogram
    if selected_hotel:
        histogram_filtered_data = df[df['name'] == selected_hotel]
        histogram_data = histogram_filtered_data['average.rating'].astype(float)
    else:
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
    img_buffer = BytesIO()
    plt.axis('off')
    plt.savefig(img_buffer, format='png')
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    return render_template('userDashboard.html', img_data=img_data, hotelNames= hotel_name, histogram_heading=histogram_heading,                        wordcloud_heading=wordcloud_heading, histogram_div=histogram_div, hotelName=selected_hotel, pie_chart_div=pie_chart_div)

def map():
    df['reviews.total'] = df['reviews.total'].astype(str)
    # Merge the data based on 'Province' and calculate the total number of reviews
    merged_data = gdf.merge(df, left_on='STUSPS', right_on='province', how='left')

    # Normalize the data if needed
    # Here, we are assuming you have a column 'Total Reviews' in your CSV data
    # You may want to scale the data to fit the color scale properly

    # Create a chloropleth map using Plotly Express
    fig = px.choropleth(merged_data, 
                        geojson=gdf.geometry, 
                        locations=merged_data.index, 
                        color='reviews.total',
                        hover_name='STUSPS')

    # Convert the map to HTML
    map_div = fig.to_html(full_html=False)
    return render_template('testfaz.html', map_div=map_div)

@app.route('/', methods=['GET', 'POST'])
def navigation():
    pie_chart_div = piechart()
    #map_div = map()
    histogram_heading = "Histogram"
    wordcloud_heading = "Word Cloud"

    hotel_name = df['name'].unique()
    
    # Create a histogram figure using go.Figure
    histogram_data = df['average.rating'].astype(float)
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
    
    #return render_template('userDashboard.html', pie_chart_div=pie_chart_div, histogram_heading=histogram_heading, wordcloud_heading=wordcloud_heading)
    return render_template('userDashboard.html',  histogram_heading=histogram_heading, histogram_div=histogram_div, wordcloud_heading = wordcloud_heading, hotelNames=hotel_name, pie_chart_div=pie_chart_div, img_data=img_data)
    #return render_template('dashboard2.html', pie_chart_div=pie_chart_div, histogram_div=histogram_div)

# import csv
@app.route('/upload_file', methods=['GET', 'POST'])
def uploadFile():
	if request.method == 'POST':
	# upload file flask
		f = request.files.get('file')

		# Extracting uploaded file name
		data_filename = secure_filename(f.filename)

		f.save(os.path.join(app.config['UPLOAD_FOLDER'],
							data_filename))

		session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],
					data_filename)

		return render_template('main2.html')
	return render_template("main.html")

@app.route('/table', methods=("POST", "GET"))
def index():
    column_val = df.columns.values
    return render_template("userPref.html", column_names=column_val, row_data=list(df.values.tolist()),zip=zip)

if __name__ == "__main__":    
    app.run(debug=True)