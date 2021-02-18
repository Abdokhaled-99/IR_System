from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
stop_words.extend(["?", ".", "'", "!", "/", ";", ","])
termdict = dict()


def pos(term):
    return termdict[term]['posting']


def getFreq(term):
    return termdict[term]['freq']


def positional_indexer():
    for docid in range(1, 11):
        with open(f'D:\LEVEL4\IR\IR_DOCS/test{docid}.txt', 'r', encoding='utf-8') as doc:
            for i, term in enumerate(word_tokenize(doc.read())):
                term = lemmatizer.lemmatize(term.lower())
                if not term in termdict and term not in stop_words:
                    termdict[term] = {'freq': 1}
                    termdict[term]['posting'] = {docid: [i]}
                elif term in termdict:
                    if docid in pos(term):
                        termdict[term]['posting'][docid].append(i)
                    else:
                        termdict[term]['posting'][docid] = [i]
                    termdict[term]['freq'] += 1


positional_indexer()

inquery = input("input query:\n")
query = [lemmatizer.lemmatize(term.lower()) for term in word_tokenize(inquery)]

d = []
for term in query:
    if term in termdict:
        d.append(pos(term).keys())
        print(f"\n<{term} : {getFreq(term)}")
        for k, v in pos(term).items():
            print(f"doc{k}:{v}")
        print("===========")
    else:
        print(f"term '{term}' not found")
print(f"documents for whole query \"{inquery}\":", end=" ")
print(set(d[0]).intersection(*d))
