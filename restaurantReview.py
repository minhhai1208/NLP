import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import nltk # natural language toolkit
import en_core_web_sm
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#read the data
df = pd.read_csv("D:\Restaurant_Reviews.tsv", sep='\t')
print(df['Review'].head(10))

# cleaning the data
df['Review'] = df['Review'].str.replace('[^a-zA-Z]', ' ', regex=True).replace(r'\s+', ' ', regex=True).replace('abov', 'above', regex= True)
print(df.head())

# check null value
print(df.isna().sum())

# check the balance of the data set
plt.hist(df[df['Liked']==0]["Liked"], bins=2 , label='Not good')
plt.hist(df[df['Liked']==1]["Liked"], bins=2, label="Good")
plt.legend()
plt.savefig('fig1.png')
plt.show()

# preparing data
x_train, x_test, y_train, y_test = train_test_split(df["Review"], df['Liked'], stratify=df['Liked'], train_size=0.8)

# after stemming and removing stopwords
usual_token = TfidfVectorizer().build_tokenizer()
stemmer = nltk.stem.PorterStemmer()

def stemmed_token(doc):
    tokens = usual_token(doc)
    return [stemmer.stem(tok) for tok in tokens]

stemmed_bow = TfidfVectorizer(tokenizer=stemmed_token, token_pattern=None, stop_words=stopwords.words('English'), ngram_range=(1, 3) , max_features=5000)

stemmed_bow.fit(x_train)
stemmed_x_train = stemmed_bow.transform(x_train)
stemmed_x_test = stemmed_bow.transform(x_test)
stemmed_x_train = stemmed_x_train.toarray()
stemmed_x_test = stemmed_x_test.toarray()

print('Vocabulary size {}'.format(len(stemmed_bow.vocabulary_)))
print('Vocabulary {}' .format(stemmed_bow.get_feature_names_out()))

# build model

classtifier = GaussianNB()
classtifier.fit(stemmed_x_train, y_train)
print(classtifier.score(stemmed_x_train, y_train))
print(classtifier.score(stemmed_x_test, y_test))

y_pred = classtifier.predict(stemmed_x_test)
cm = confusion_matrix(y_test, y_pred)

# Create a ConfusionMatrixDisplay object
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not good', 'good'])

fig, ax = plt.subplots(figsize=(8, 6))  # Optional: Specify the size of the plot
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Confusion Matrix')
plt.savefig('fig2.png')
plt.show()
