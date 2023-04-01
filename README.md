## NLP MultiEUREX multilabel multiclass classification

To run code please install requirements.txt with package versions first

Main file: NLP-project-summary.ipynb

In the folders you can find file for each model type and utility functions 
needed


**Documents**

MultiEURLEX of Chalkidis et al. (2021) comprises 65k EU laws in 23 official EU languages. Each EU law has been annotated with EUROVOC concepts (labels) by the Publication Office of EU. Each EUROVOC label ID is associated with a label descriptor, e.g., [60, agri-foodstuffs], [6006, plant product], [1115, fruit]. The descriptors are also available in the 23 languages. Chalkidis et al. (2019) published a monolingual (English) version of this dataset, called EUR-LEX, comprising 57k EU laws with the originally assigned gold labels. 

The following data fields are provided for documents (train, dev, test):

_celex_id_: (str) The official ID of the document. The CELEX number is the unique identifier for all publications in both Eur-Lex and CELLAR.
_text_: (str) The full content of each document across languages.
_labels_: (List[int]) The relevant EUROVOC concepts (labels).

**Aim**</br>
The aim of this project is to predict document classification using MultiEURLEX dataset from huggingface, which includes over 23 languages, and we chose 3 languages. We initially wanted to take 3 languages from three language families but decided to stick with _English_, _French_, and _Spanish_. 
We use a non-ML baseline model, an LSTM model, and a BERT model for the classification task.

**Data**</br>
French and English had the highest dataset at 55k, followed by the Spanish at 23k. We split it into train, validation and test. 

**Preprocessing**</br>
We preprocessed the data by performing the following operations:</br>
- [x] Remove the punctuation and stopwords</br>
- [x] make all text lowercase</br>
- [x] tokenize</br>
- [x] stem and lemmatize</br>

**Models**</br>
For mor answers on the models, please visit [__NLP-project-summary.ipynb__](https://github.com/jon-robbins/nlp-final/blob/main/NLP-project-summary.ipynb) and also we have several notebooks with the RNN and BERT model. 