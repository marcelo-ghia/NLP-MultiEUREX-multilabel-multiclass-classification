import numpy as np
import pandas as pd
import nltk
import re
nltk.download('stopwords')

nltk.download('punkt')
#nltk.download('stopwords')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')

test_data = pd.DataFrame([
    {'id': 1234, 'text': 'This is a sentence'},
    {'id': 1432, 'text': 'This is also a sentence'},
    {'id': 987152, 'text': 'This is not a sentence'},
    {'id': 5235, 'text': 'This is a book'},
    {'id': 12361624, 'text': 'This is a collection of words'},
])

def preprocess_text(text):
    """
    Apply this fucntion to text to:
    1. lowercase
    2. remove punctuation
    3. tokenize
    4. remove stopwords
    5. stem
    6. lemmatize
    """
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

def get_stopwords(df, text_col, id_col, stopwords=stopwords, threshold=0.5):
    """
    This function will take a dataframe and its text, and return the number of documents that contain each individual word. 
    This is useful for removing stopwords from your documents. A common threshold to use is 0.5 (if a word appears in >=50% of your
    documents then it's probably not useful for classification).  

    This will return two items: the most common words, and a list of words within your given threshold
    """
    word_counts = df.set_index(id_col)[text_col].str.split()\
        .explode()\
        .reset_index()\
        .drop_duplicates()\
        .groupby(text_col).size()\
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

def extract_date(text):
    """
    This function when applied to the original text of a document will extract its publishing date and return a pd.Datetime object
    """
    pattern = r'(?<=of\s)(\d{1,2})\s([A-Za-z]+)\s(\d{4})'
    match = re.search(pattern, text.lower())
    if match:
        date_str = match.group()
        return pd.to_datetime(date_str, format='%d %B %Y')
    else:
        return np.nan
    
def get_eu_legal_type(text):
    """
    This will extract the type of legal document it is. Apply this function to the dataframe and create a new column
    """
    text = text.lower()
    

    regex_patterns = [
        r"(directive)(.*)(of the european parliament and of the council)",
        r"(decision)(.*)(of the european parliament and of the council)",
        r"(decision)(.*)(of the european council)",
        r"(decision)(.*)(of the european council)",
        r"(regulation)(.*)(of the european parliament and of the council)",
        r"(decision)(.*)(of the european parliament and of the council)",
        r"(directive)(.*)(of the european parliament and of the council)",
        r"(regulation)(.*)(of the council)",

    ]

    for pattern in regex_patterns:
        match = re.search(pattern, text)
        if match:
            return f"{match.group(1).title()} {match.group(3).title()}"
    
    legal_types = ["commission regulation", "commission decision", "council regulation",
                "council directive", "council decision", "commission implementing regulation",
                "commission delegated regulation", 'decision of the council and the commission', 'political and security committee decision', 'european parliament decision',
                'decision of the european parliament', 'commission directive', 'decision of the council', 'decision of the european central bank', 'council implementing decision',]
    regex = r"\b(" + "|".join(legal_types) + r")\b"
    match = re.search(regex, text)
    if match:
        return match.group(1).title()
    else:
        return np.nan

def binarize_doc_type(df, doc_type_col='doc_type'):
    """
    This should be used in conjunction with the get_eu_legal_type function. First get the title with that function,
    then use this to match it to the doc. 
    
    Documents have different titles indicating the type of document (regulation, decision, directive), and who 
    published them (parliament, committee, council). This function will add binary features to determine
    who published it, and what kind of document it is. 
    """
    doc_type_map = [
    {'title': 'Commission Regulation', 'commission': 1, 'regulation': 1},
    {'title': 'Commission Decision', 'commission': 1, 'decision': 1},
    {'title': 'Council Regulation', 'council': 1, 'regulation': 1},
    {'title': 'Council Decision', 'council': 1, 'decision': 1},
    {'title': 'Directive Of The European Parliament And Of The Council', 'directive': 1, 'parliament': 1, 'council': 1},
    {'title': 'Council Directive', 'directive': 1, 'council': 1},
    {'title': 'Regulation Of The European Parliament And Of The Council', 'regulation': 1, 'parliament': 1, 'council': 1},
    {'title': 'Commission Directive', 'directive': 1, 'commission': 1},
    {'title': 'Regulation of the Council', 'regulation': 1, 'council': 1},
    {'title': 'Decision Of The European Parliament And Of The Council', 'decision': 1, 'council': 1, 'parliament': 1},
    {'title': 'Decision Of The European Parliament', 'decision': 1, 'parliament': 1},
    {'title': 'Political And Security Committee Decision', 'decision': 1, 'committee': 1},
    {'title': 'European Parliament Decision', 'decision': 1, 'parliament': 1},
    {'title': 'Decision Of The Council And The Commission', 'decision': 1, 'council': 1, 'commission': 1},
    {'title': 'Decision Of The Council', 'decision': 1, 'council': 1},
    {'title': 'Decision Of The European Council', 'decision': 1, 'council': 1},
    {'title': 'Council Implementing Decision', 'decision': 1, 'council': 1},
    {'title': 'Decision Of The European Central Bank', 'decision': 1}
]

    # Create DataFrame from list of dictionaries
    doc_types = pd.DataFrame(doc_type_map)

    # Replace missing values with 0
    doc_types = doc_types.fillna(0)
    merged_df = pd.merge(df, doc_types, left_on=doc_type_col, right_on='title')
    return merged_df

def remove_common_words(df, text_col, id_col, threshold=0.5):
    """
    This function will take a dataframe and its text and remove words that appear in the percentage of documents that you set in the threshold. Default is 0.5.
    """
    
    stopwords = set(nltk.corpus.stopwords.words('english'))

    word_counts = df.set_index(id_col)[text_col].str.split()\
        .explode()\
        .reset_index()\
        .drop_duplicates()\
        .groupby(text_col).size()\
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
    stopwords.update(frequent_words)
    def remove_stopwords(text, stopwords=stopwords):
        words = text.split()
        words = [word for word in words if word not in stopwords]
        text = ' '.join(words)
        return text
    df[text_col] = df[text_col].apply(remove_stopwords)
    return df

def preprocess_text_fr(text):
    """
    Apply this fucntion to text to:
    1. lowercase
    2. remove punctuation
    3. tokenize
    4. remove stopwords
    5. stem
    6. lemmatize
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers and punctuation using regex
    text = re.sub(r'[^\w\s]', '', re.sub(r'\d+', '', text))

    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stop words
    stopwords = nltk.corpus.stopwords.words('french')
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

def remove_common_words_fr(df, text_col, id_col, threshold=0.5):
    """
    This function will take a dataframe and its text and remove words that appear in the percentage of documents that you set in the threshold. Default is 0.5.
    """
    
    stopwords = set(nltk.corpus.stopwords.words('french'))

    word_counts = df.set_index(id_col)[text_col].str.split()\
        .explode()\
        .reset_index()\
        .drop_duplicates()\
        .groupby(text_col).size()\
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
    stopwords.update(frequent_words)
    def remove_stopwords(text, stopwords=stopwords):
        words = text.split()
        words = [word for word in words if word not in stopwords]
        text = ' '.join(words)
        return text
    df[text_col] = df[text_col].apply(remove_stopwords)
    return df
def preprocess_text_sp(text):
    """
    Apply this fucntion to text to:
    1. lowercase
    2. remove punctuation
    3. tokenize
    4. remove stopwords
    5. stem
    6. lemmatize
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers and punctuation using regex
    text = re.sub(r'[^\w\s]', '', re.sub(r'\d+', '', text))

    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stop words
    stopwords = nltk.corpus.stopwords.words('spanish')
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

def remove_common_words_sp(df, text_col, id_col, threshold=0.5):
    """
    This function will take a dataframe and its text and remove words that appear in the percentage of documents that you set in the threshold. Default is 0.5.
    """
    
    stopwords = set(nltk.corpus.stopwords.words('spanish'))

    word_counts = df.set_index(id_col)[text_col].str.split()\
        .explode()\
        .reset_index()\
        .drop_duplicates()\
        .groupby(text_col).size()\
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
    stopwords.update(frequent_words)
    def remove_stopwords(text, stopwords=stopwords):
        words = text.split()
        words = [word for word in words if word not in stopwords]
        text = ' '.join(words)
        return text
    df[text_col] = df[text_col].apply(remove_stopwords)
    return df