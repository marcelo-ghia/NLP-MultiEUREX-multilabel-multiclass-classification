## NLP MultiEUREX multilabel multiclass classification

To run code please install requirements.txt with package versions first

Main file: NLP-project-summary.ipynb

In the folders you can find files for each model type and utility functions 
needed. Our shared library for preprocessing functions are located in the main directory. 


Documents

MultiEURLEX of Chalkidis et al. (2021) comprises 65k EU laws in 23 official EU languages. Each EU law has been annotated with EUROVOC concepts (labels) by the Publication Office of EU. Each EUROVOC label ID is associated with a label descriptor, e.g., [60, agri-foodstuffs], [6006, plant product], [1115, fruit]. The descriptors are also available in the 23 languages. Chalkidis et al. (2019) published a monolingual (English) version of this dataset, called EUR-LEX, comprising 57k EU laws with the originally assigned gold labels.
Aim
The aim of this project is to predict document classification using MultiEURLEX dataset from huggingface. 

The following data fields are provided for documents (train, dev, test):

celex_id: (str) The official ID of the document. The CELEX number is the unique identifier for all publications in both Eur-Lex and CELLAR.
text: (str) The full content of each document across languages.
labels: (List[int]) The relevant EUROVOC concepts (labels).



