import json
import plotly
import pandas as pd
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import pickle
import sys

# Import for wrangling data to produce graphs
sys.path.append("../wrangling")
import wrangling

sys.path.append("../models")
# Import functions used to create the classifier model
from train_classifier import *

# Specify the path with static images
app = Flask(__name__,static_url_path='/static')
    
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorised_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    #Graph 1
    genre_counts,genre_names = wrangling.get_category_counts(df)

    #Graph 2
    average_lengths = wrangling.get_avg_lengths(df,genre_names)

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Number of messages per category',
                'yaxis': {
                    'title': "number of messages"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

                {
            'data': [
                Bar(
                    # Sort by average length ascending
                    x=[name for length, name in sorted(zip(average_lengths,genre_names))],
                    y=[length for length, name in sorted(zip(average_lengths,genre_names))],
                )
            ],

            'layout': {
                'title': 'Average number of words used per message in category',
                'yaxis': {
                    'title': "number of words",
                    'range': [0, 70]
                },
                'xaxis': {
                    'title': ""
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, genre_names=genre_names, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

@app.route('/select')
def select():

    # save user input in query
    category = request.args.get('category', '') 
    message = f"user submitted {category}"

    try : # try reading the clean words from a file
        clean_df = pd.read_csv('clean.csv')
    except: # if there is no file create one
        clean_df = wrangling.clean_message_col(df)

    word_list = wrangling.get_catwordlist(clean_df,input_category=category)
    topn = wrangling.topn_words(word_list=word_list,n=10)
    data = {'word': list(topn.keys()), 'frequency': list(topn.values())}
    data = pd.DataFrame.from_dict(data)
    data=data.to_html(index=False)
    data=data.replace('<th>','<th style="text-align:center">')
    data=data.replace('<table','<table align="center"')
    data=data.replace('<tr>', '<tr align="center">')

    return render_template(
        'select.html',
        category=category,
        topn=data
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()