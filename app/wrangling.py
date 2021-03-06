import numpy as np

"""Graph 1 : message genres"""
def get_category_counts(df):
    features = list(df.columns[4:])
    X = df['message']
    Y = df[features]
    cat_counts = Y.sum().sort_values(ascending=True)
    genre_counts = cat_counts.values
    genre_names = list(cat_counts.index)
    return genre_counts,genre_names

"""Graph 2 : message lengths"""
def get_avg_lengths(df,features): 
    """
    This function obtains average number of words per message across 
    different message categories. 
    
    args:
        df: pandas dataframe with a text column and a category column
        features: binary columns indicating the category.
    """
    # Get message length feature
    df['no_words'] = df['message'].apply(lambda x: len(x.split())).values
    
    # Get a list of average lengths for each of the categories
    average_lengths=[]
    for cat in features:
        length = df.loc[df[cat]==1]['no_words']
        average_lengths.append(np.mean(length))

    # Fill na in the list 
    average_lengths = [int(x) if x>0 else 0 for x in average_lengths]
    
    # Ensure the number of averages equals number of categories
    assert len(average_lengths)==len(features)
    
    #Get rid of the column
    df.drop('no_words', axis=1, inplace=True)

    return average_lengths

import sys
sys.path.append("../models")
# Import functions used to create the classifier model
from train_classifier import *

"""---------------------Top words------------------------"""
# Helper function to clean a df
def clean_message(text,lemmatizer = WordNetLemmatizer()):
    """Clean words"""
    url_marked = mark_urls(text)
    reduced_length = reduce_length(url_marked)
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", reduced_length))

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    clean_tokens = [t for t in clean_tokens if t not in set(stopwords.words('english'))]
    clean_message = " ".join(clean_tokens)
    return clean_message

def clean_message_col(df):
    """
    Function to clean messages dataframe
    """
    print('cleaning messages...')
    df_clean_messages = df.copy(deep=True)
    df_clean_messages['message'] = df_clean_messages['message'].apply(
        lambda x: clean_message(x))
    print('messages clean')
    print(df_clean_messages['message'])
    df_clean_messages.to_csv('clean.csv')

def get_catwordlist(df,input_category):
    """
    Function to filter dataframe for selected category and exract 
    word list
    """
    cat_selected = df.loc[df[input_category]==1]['message'].fillna('empty')
    cat_selected=list(cat_selected)
    cat_vocab = []
    for message in cat_selected:
        cat_vocab.append(message.split())
    word_list = [item for sublist in cat_vocab for item in sublist]
    return word_list

def topn_words(word_list,n):
    """
    Get top 5 words from a list of words
    """
    # Calculate frequency distribution
    fdist = nltk.FreqDist(word_list)
    freq = {}

    # Output top n words
    for word, frequency in fdist.most_common(n):
        freq[word]=frequency

    return {k: freq[k] for k in list(freq)[:n]}

if __name__ == "__main__":
    category='related'
    clean_df = pd.read_csv('clean.csv')
    clean_df = clean_df.dropna(subset=['message'])
    print(clean_df['message'].head())
    word_list = get_catwordlist(clean_df,input_category=category)
    topn = topn_words(word_list=word_list,n=5)
    print(topn)
