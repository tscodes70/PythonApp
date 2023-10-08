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
df = pd.read_csv('NEWOUTPUT.csv', index_col=0)

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
    top_categories = df['Category'].value_counts().nlargest(4)
    # Create the Plotly pie chart
    pie_chart = go.Figure(data=[go.Pie(labels=top_categories.index, values=top_categories)])
    # Convert the chart to an HTML div
    pie_chart_div = pie_chart.to_html(full_html=False)

    return pie_chart_div
    
def histogram():
    # histogram
    ratingColumn = 'Average Rating'
    histogram = px.histogram(df, x=ratingColumn, title=f'Histogram of Average Rating')
    histogram_div = histogram.to_html()
    
    return histogram_div

@app.route('/', methods=['GET', 'POST'])
def navigation():
    pie_chart_div = piechart()
    histogram_div = histogram()

    hotel_name = df['Hotel Name'].unique()

    selected_province = request.args.get('province')

    if selected_province:
        filtered_data = df[df['Province'] == selected_province]
        text = ' '.join(filtered_data['Review Summary'].astype(str))
    else:
        text = ' '.join(df['Review Summary'].astype(str))
        
    # Replace 'column_name' with the name of the CSV column you want to create a word cloud from
    # text = ' '.join(df['Review Summary'].astype(str))

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
    
    return render_template('userDashboard.html', hotelNames=hotel_name, pie_chart_div=pie_chart_div, histogram_div=histogram_div, img_data=img_data)
    #return render_template('dashboard2.html', pie_chart_div=pie_chart_div, histogram_div=histogram_div)

@app.route('/wordcloud', methods=['POST'])
def wordcloud():
    selected_hotel = request.form['hotelName-dropdown']

    if selected_hotel:
        filtered_data = df[df['Hotel Name'] == selected_hotel]
        text = ' '.join(filtered_data['Review Summary'].astype(str))
    else:
        text = ' '.join(df['Review Summary'].astype(str))
    
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

    return render_template('userDashboard.html', img_data=img_data, hotelName=selected_hotel)

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