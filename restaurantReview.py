import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import nltk # natural language toolkit
import en_core_web_sm
from nltk.corpus import stopwords
#read the data
df = pd.read_csv("D:\Restaurant_Reviews.tsv", sep='\t')
print(df['Review'].head(10))

# cleaning the data
df['Review'] = df['Review'].str.replace('[^a-zA-Z]', ' ', regex=True).replace(r'\s+', ' ', regex=True)
print(df.head())

# after stemming and removing stopwords
usual_token = TfidfVectorizer().build_tokenizer()
stemmer = nltk.stem.PorterStemmer()

def stemmed_token(doc):
    tokens = usual_token(doc)
    return [stemmer.stem(tok) for tok in tokens]

stemmed_bow = TfidfVectorizer(tokenizer=stemmed_token, token_pattern=None, stop_words=stopwords.words('English'))

stemmed_bow.fit(df['Review'])
stemmed = stemmed_bow.transform(df['Review'])

print('Vocabulary size {}'.format(len(stemmed_bow.vocabulary_)))
print('Vocabulary {}' .format(stemmed_bow.get_feature_names_out()))



