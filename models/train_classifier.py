# ETL
import pandas as pd
import numpy as np
import sqlite3
import sys
import re

# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['stopwords','punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')

# ML
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Multiclass
import pickle
import sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import metrics

# EDA
import matplotlib.pyplot as plt

def load_data(database_filepath):
    # load data from database
    conn = sqlite3.connect(database_filepath)
    cur = conn.cursor()
    sql = "select * from categorised_messages"
    df = pd.read_sql(sql, conn)
    category_names = list(df.columns[4:])
    X = df['message']
    Y = df[category_names]
    return X, Y, category_names

def mark_urls(text):
    """ Helper function to replace urls in text with placeholders"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    return text

def reduce_length(text):
    """ Helper function to remove more than 2 characters of the 
    same kind occuring one after another"""
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def cleaner_tokenizer(text,lemmatizer = WordNetLemmatizer()):
    """Main function to clean and tokenize text"""
    url_marked = mark_urls(text)
    reduced_length = reduce_length(url_marked)
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", reduced_length))

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    clean_tokens = [t for t in clean_tokens if t not in set(stopwords.words('english'))]
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """ A class that uses sentence tokens and parts of speach tagger to extract if the first
    is a verb in base or present form. The other forms should be handled by lemmatizer.
    
        target tags:
            VB (verb), base form, ex. help
            VBP (verb), sing. present, ex.help
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(cleaner_tokenizer(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP']:
                    return 1
                else:
                    return 0
            except: return 0
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(lambda x: self.starting_verb(x)).fillna(0).values
        return pd.DataFrame(X_tagged)
    
def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=cleaner_tokenizer,max_features=3000)),
                ('tfidf', TfidfTransformer())
            ])),
            ('startverb', StartingVerbExtractor())
            ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])
    return pipeline

def evaluate_model(model, X_test, Y_test,verbose=True):
    """Function to evaluate the performance of multiclass model.

    Args:
        model: classifier model
        X_test: features to predict from
        Y_test: True values to be predicted
        
        verbose: if True: see the performance across each of the categories
        if False: show only a weighted F1 score.
    """
    predicted = model.predict(X_test)
    f1_scores = []
    for index, col in enumerate(Y_test.columns):
        if verbose:
            print(col)
            print(classification_report(Y_test.iloc[:, index],[row[index] 
                for row in predicted]))
            print('---------------------------------------------------')
        score = f1_score(Y_test.iloc[:, index],  
            [row[index] for row in predicted],
            average='weighted')
        f1_scores.append(score)
    avg_f1 = np.mean(f1_scores)
    print(f'Avg weighted f1-score:{round(avg_f1,3)}')

def save_model(model, model_filepath):
    with open('models/classifier.pkl', 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()