import numpy as np
import pandas as pd
import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')

test_data = pd.DataFrame([
    {'id': 1234, 'text': 'This is a sentence'},
    {'id': 1432, 'text': 'This is also a sentence'},
    {'id': 987152, 'text': 'This is not a sentence'},
    {'id': 5235, 'text': 'This is a book'},
    {'id': 12361624, 'text': 'This is a collection of words'},
])

def get_stopwords(df:pd.DataFrame, text_col:str, id_col:str, stopwords=stopwords:set, threshold:float):
    """
    This function will take a dataframe and its text, and return the number of documents that contain each individual word. 
    This is useful for removing stopwords from your documents. A common threshold to use is 0.5 (if a word appears in >=50% of your
    documents then it's probably not useful for classification).  
    """
    word_counts = df.set_index(id_col)[text_col].str.split()\
        .explode()\
        .reset_index()\
        .drop_duplicates()\
        .groupby('pp_text').size()\
        .sort_values(ascending=False)
    word_counts = word_counts.to_dict()
    word_counts_df = []
    for i, v in word_counts.items():
        result = {}
        result['word'] = i
        result['count'] = v
        word_counts_df.append(result)
    word_counts_df = pd.DataFrame(word_counts_df).sort_values(by='count',ascending=False)
    word_counts_df['pct'] = word_counts_df['count']/len(df)
    frequent_words = word_counts_df.loc[word_counts_df['pct'] >= threshold, 'word'].tolist()
    return word_counts_df, frequent_words

counts, to_add = get_stopwords(test_data, text_col='text', id_col='id', stopwords=stopwords, threshold=0.5)

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers and punctuation using regex
    text = re.sub(r'[^\w\s]', '', re.sub(r'\d+', '', text))

    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stop words
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_words = [word for word in words if word not in stopwords]

    # Stem the remaining words using Porter Stemmer
    stemmer = nltk.PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Lemmatize the remaining words using WordNetLemmatizer
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

    # Join the preprocessed words into a single string
    preprocessed_text = " ".join(lemmatized_words)

    return preprocessed_text