# Disaster Response Pipeline Project

* <b>Summary</b>: 
I have analysed the disaster data from <a href="https://www.figure-eight.com/">Figure Eight</a> and built a Random Forest model for an API that classifies disaster messages across 36 categories. The model performs relatively well- <b>avg weighted f1-score:0.94</b>. 

* <b>Purpose</b>: gain experience in writing Data Engineering Pipelines, Machine Learning Pipelines and web development with Flask.

* <b>Task</b>: Create a multiclass model predicting the emergency categories that a message may belong to.

## Project Components:

1. <b>ETL Pipeline</b>- process_data.py

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

2. <b>ML Pipeline</b>- train_classifier.py

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

3. <b>Flask Web App</b>- run.py
* Classifies inputed message using the pickle model
* Includes 2 interactive visualisations
* Query specific category for top words

## Project structure:

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
| - static # Folder with static data visualisations
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

## Requirements:

```
python==3.7.6
Flask==1.1.2
matplotlib==3.1.3
numpy==1.18.1
pandas==1.0.2
pickleshare==0.7.5
plotly==4.6.0
scikit-learn==0.22.1
SQLAlchemy==1.3.16
wordcloud==1.6.0
sys
re
json
```

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Further recommendations:
To further improve the model, I recommend more data cleaning as well as adding word to vec feature embeddings. I would also try to reduce the class imbalance and see if it can improve the model performance.

## Credits:
Thanks to Udacity and Figure Eight for providing the project idea and data to work with.
