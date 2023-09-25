from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename
import subprocess
import plotly.graph_objs as go

from flask import Flask, render_template
# import analyze

UPLOAD_FOLDER = os.path.join('dataset', 'uploads')

# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'This is your secret key to utilize session in Flask'

# upload the file into the local folder
@app.route('/', methods=['GET', 'POST'])
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

# show raw data on webpage when user clicks on the 'show data' button 
@app.route('/show_data')
def showData():
	# Uploaded File Path
	data_file_path = session.get('uploaded_data_file_path', None)
	# read csv
	uploaded_df = pd.read_csv(data_file_path,
							encoding='unicode_escape')
	# Converting to html Table
	uploaded_df_html = uploaded_df.to_html()
	return render_template('show_csv.html',
						data_var=uploaded_df_html)
 
# run analyze.py
@app.route('/analyze')
def analyzaData():
    subprocess.run(['python', 'analyze.py'])
    return redirect(url_for('categoriseData'))

# format the "Category" column of the csv file
@app.route('/categorise_data')
def categoriseData():
	# Load the original CSV file
    input_csv_file = 'outputdata.csv'  
    output_csv_file = 'NEWOUTPUT.csv'  # Replace with the path to your output CSV file

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(input_csv_file)

    # Specify the column that contains the values to process
    target_column = 'Category'  
    
    # Replace "and" and "&" with a comma and strip leading/trailing spaces
    df[target_column] = df[target_column].str.replace(r' and | & ', ',' ,regex=True)

    # remove spaces
    df[target_column] = df[target_column].str.replace(r'\s', '', regex=True)

    # remove the starting spaces
    df[target_column] = df[target_column].str.lstrip()
    
    # Convert all text in the specified column to lowercase
    df[target_column] = df[target_column].str.lower()
    
    # Create a list to store the rows with split values
    new_rows = []

    # Iterate through the DataFrame
    for index, row in df.iterrows():
        # Split the value in the column by a comma (,)
        split_values = row['Category'].split(',')

        # Create new rows for each split value
        for value in split_values:
            new_row = row.copy()  # Copy the original row
            new_row['Category'] = value  # Update the column with the split value
            new_rows.append(new_row)

    # Create a new DataFrame with the new rows
    new_df = pd.DataFrame(new_rows)
    
    # replace any string with "hotel" with hotels
    new_df[target_column] = new_df[target_column].apply(lambda x: 'motels' if 'motel' in x.lower() else x)
    new_df[target_column] = new_df[target_column].apply(lambda x: 'hotels' if 'hotel' in x.lower() else x)
    new_df[target_column] = new_df[target_column].apply(lambda x: 'lodgings' if 'lodging' in x.lower() else x)
    
    # Use the drop_duplicates() method to remove duplicate rows
    new_df = new_df.drop_duplicates()
    
    # Write the new DataFrame to a new CSV file
    new_df.to_csv(output_csv_file, index=False)
    
    return redirect(url_for('dataVisualisation'))

@app.route('/data_visualisation')
def dataVisualisation():
	# Load the CSV data into a Pandas DataFrame
    df = pd.read_csv('NEWOUTPUT.csv')

    # Assuming the CSV has a 'Category' column, get the top 4 values
    top_categories = df['Category'].value_counts().nlargest(4)

    # Create the Plotly pie chart
    pie_chart = go.Figure(data=[go.Pie(labels=top_categories.index, values=top_categories)])

    # Convert the chart to an HTML div
    pie_chart_div = pie_chart.to_html(full_html=False)

    return render_template('dashboard.html', pie_chart_div=pie_chart_div)

if __name__ == '__main__':
	app.run(debug=True)
