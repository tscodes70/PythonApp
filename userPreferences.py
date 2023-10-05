from flask import Flask, render_template, request, session, redirect
from flask_bootstrap import Bootstrap
import pandas as pd
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

def openCsv():
    # takes data from outputdata.csv and loads it into a pandas dataframe
    df = pd.read_csv('outputdata.csv', index_col=0)
    return df

@app.route('/', methods=("POST", "GET"))
def index():
    df = openCsv()
    column_val = df.columns.values
    return render_template("userPref.html", column_names=column_val, row_data=list(df.values.tolist()),zip=zip)

if __name__ == "__main__":    
    app.run(debug=True)