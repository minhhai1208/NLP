import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC

#lean data
df = pd.read_csv("D:\\spam_ham_dataset.csv")

print(df)
print("Check null values")
print(df.isna().sum())

print("Check duplicated values")
print(df.duplicated().sum())
df.drop_duplicates(keep = 'first', inplace=True)

print(df['label'].value_counts()/len(df))

df.drop(['Unnamed: 0', 'label'], axis = 1, inplace=True)
df.rename(columns={'label_num':'label'}, inplace=True)

plt.hist(df[df['label'] == 0]['label'], bins = 2, alpha = 0.7, label='pam')
plt.hist(df[df['label'] == 1]['label'], bins = 2, alpha = 0.7, label='Spam')
plt.legend()
plt.savefig('fig1.png')
train_data = df['text']
label = df['label']
x_train, x_test, y_train, y_test = train_test_split(train_data,label, stratify=label, train_size=0.8)

#preprocessing
vect = TfidfVectorizer()
vect.fit(x_train)
tf_data = vect.transform(x_train)
tf_test_data = vect.transform(x_test)

svc = SVC(kernel= "sigmoid", gamma  = 1.0)

svc.fit(tf_data, y_train)
print("Train accuracy:", svc.score(tf_data, y_train))
print('Test accuracy' , svc.score(tf_test_data, y_test))

y_pred = svc.predict(tf_test_data)
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a ConfusionMatrixDisplay object
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pam', 'Spam'])

fig, ax = plt.subplots(figsize=(8, 6))  # Optional: Specify the size of the plot
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Confusion Matrix')

# Save the plot to a file
plt.savefig('confusion_matrix.png', dpi=300)

plt.show()
