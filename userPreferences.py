from flask import Flask, render_template, request, session, redirect
from flask_bootstrap import Bootstrap
import pandas as pd
#import numpy as np

app = Flask(__name__)
Bootstrap(app)

def main():
    print("hello world!")
    return

def openCsv():
    return

# test rendering get better formats next time
@app.route('/', methods=("POST", "GET"))
def index():
    df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
                   'B': [5, 6, 7, 8, 9],
                   'C': ['a', 'b', 'c--', 'd', 'e']})
    return render_template("userPref.html", tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == "__main__":    
    app.run(debug=True)

main()