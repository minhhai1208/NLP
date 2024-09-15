import spacy
import en_core_web_md
import en_core_web_sm
import nltk
from nltk.stem.porter import PorterStemmer
from nltk .stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from spacy.matcher import Matcher

s1 = 'Appel is looking at buying U.K startup for $1 billion !'
words = ['runs', 'ran' , 'played', 'beautifully']
words1 = ['are', 'is' , 'was' , 'will do']

# tokenization
nlp = en_core_web_sm.load()

# Stemming
p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer(language= 'english')

for word in words :
    print(word + '-----' + p_stemmer.stem(word))
    print(word + '-----' +s_stemmer.stem(word))

print(s_stemmer.stem(s1))

# lemma
lemma = WordNetLemmatizer()

for word in words1:
    print(f"{word} -> {lemma.lemmatize(word, pos=wordnet.VERB)}")

#Vocabulary and Matching
matcher = Matcher(nlp.vocab)

speech = "David and Emma looked at each other across the table."


pattern1 = [{'LOWER': 'emma'}, {'POS': 'VERB'}]  # Matches verbs in base form


matcher.add('Emma', [pattern1])

doc = nlp(speech)
for token in doc:
    print(token)
find_match = matcher(doc)
print(find_match)



