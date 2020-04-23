import sys
# ETL
import pandas as pd
import sqlite3
import re

# NLTK
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ML
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Multiclass
import sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import f1_score
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import pickle

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

class Text_Length(BaseEstimator, TransformerMixin):
    """A class that gets a text length from text"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(lambda x: len(x)).values
        return pd.DataFrame(X_tagged)
    
def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=cleaner_tokenizer,ngram_range=(1, 1))),
                ('tfidf', TfidfTransformer())
            ])),

            ('txt_len', Text_Length())
        ])),

        ('clf', RandomForestClassifier(n_jobs=-1))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names,verbose=False):
    predicted = model.predict(X_test)
    f1_scores = []
    for index, col in enumerate(Y_test.columns):
        if verbose:
            print(classification_report(Y_test.iloc[:, index], [row[index] for row in predicted]))
        score = f1_score(Y_test.iloc[:, index],  
            [row[index] for row in predicted],
            average='weighted')
        f1_scores.append(score)
    avg_f1 = np.mean(f1_scores)
    print(f'Avg weighted f1-score:{round(avg_f1,2)}')

def save_model(model, model_filepath):
    with open('classifier.pkl', 'wb') as file:
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
        evaluate_model(model, X_test, Y_test, category_names)

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