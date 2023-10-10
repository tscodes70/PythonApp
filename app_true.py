from flask import Flask, render_template, request, session
from flask_bootstrap import Bootstrap
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
#import numpy as np
from json import dumps,loads
import globalVar

'''
TODO
> all file address to be resolved

> current flow FOR UI
- upload raw csv to dataset(DONE), get name of hotel of user(NOT ADDED) 
- processing/cleaning/analysis in the backend (NOT ADDED)
- display info after cleaning
-- Home : general info with categories, amenities, word map for given hotel/ pir chart etc,etc.(BASE DONE, WORD MAP NOT ADDED, NOT YET CATERED TO USER)
-- Sentiment Analysis : More additional info of analysis of the hotel review, compound review, prediction score etc,etc. (NOT ADDED)
-- Compare : Compare stats of hotel to similar hotels in budget ranking (NOT ADDED)
-- View Summary : view all cleaned hotel info in a table with search, can export this data(VIEW TABLE DONE, EXPORT NOT ADDED)
-- Upload new Dataset : go back to start to get a new dataset
'''

app = Flask(__name__)
Bootstrap(app)

#ask user to upload csv_file first
@app.route('/', methods=['GET','POST'])
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

	return render_template("new_upload.html")

@app.route('/dashboard', methods=['GET','POST'])
def navigation():
    # Load the CSV data into a Pandas DataFrame
    df = pd.read_csv('../csvs/analyzedhotels_10-Oct.csv')
    columnhead = df['name'].tolist()
    # Assuming the CSV has a 'Category' column, get the top 4 values
    top_categories = df[globalVar.CATEGORIES].value_counts().nlargest(4)

    # Create the Plotly pie chart
    pie_chart = go.Figure(data=[go.Pie(labels=top_categories.index, values=top_categories)])

    # Convert the chart to an HTML div
    pie_chart_div = pie_chart.to_html(full_html=False)
    
    ratingColumn = globalVar.AVERAGE_RATING
    histogram = px.histogram(df, x=ratingColumn, title=f'Histogram of Average Rating')
    histogram_div = histogram.to_html()

    #PLACEHOLDER FOR GENERAL OVERVIEW GRAPHS
    pie_chart_div2 = pie_chart.to_html(full_html=False)
    histogram_div2 = histogram.to_html()

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

    return render_template('userDashboard.html', columnhead=columnhead,pie_chart_div=pie_chart_div, histogram_div=histogram_div, 
                           histogram_div2=histogram_div2, pie_chart_div2=pie_chart_div2,correlationgraph=correlationgraph)

def openCsv():
    # takes data from [FROM read data] and loads it into a pandas dataframe
    df = pd.read_csv('../csvs/analyzedhotels_10-Oct.csv', index_col=0)
    df.columns = df.columns.str.replace('.', '_')
    return df

@app.route('/api/general')
def summary():
     df = openCsv()
     jsonfile = df.to_json(orient='table')
     return jsonfile

@app.route('/table', methods=("POST", "GET"))
def index():
    df = openCsv()
    #column_val = df.columns.values
    #return render_template("viewSummary.html", column_names=column_val, row_data=list(df.values.tolist()),zip=zip)
    return render_template("viewSummary.html", title="View Summary")

if __name__ == "__main__":    
    app.run(debug=True)


