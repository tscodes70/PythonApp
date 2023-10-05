from flask import Flask, render_template, request, session, redirect
from flask_bootstrap import Bootstrap
import pandas as pd
#import numpy as np

app = Flask(__name__)
Bootstrap(app)

def openCsv():
    df = pd.read_csv('outputdata.csv', index_col=0)
    return df

# test rendering get better formats next time
@app.route('/', methods=("POST", "GET"))
def index():
    df = openCsv()
    #new_table = df.drop('Unnamed: 0', axis=1)
    column_val = df.columns.values
    return render_template("userPref.html", column_names=column_val, row_data=list(df.values.tolist()),
                           link_column='Hotel Name', zip=zip)

if __name__ == "__main__":    
    app.run(debug=True)