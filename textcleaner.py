import re
import pickle
from nltk.stem import WordNetLemmatizer, PorterStemmer 


def transform_text(string):
    lem = WordNetLemmatizer()
    bagW = []
    #regex the string
    new_1 = re.sub('(http://.*?\s)|(http://.*)',' ',str(string))
    new_2 = re.sub('[.*_]','',new_1)
    new_3 = re.sub('\d','',new_2)
    new_4 = re.sub('\W',' ',new_3)
    
    #lemmatize the string
    liner = new_4.lower().split()
    for word in liner:
        a = lem.lemmatize(word, "v")
        b = lem.lemmatize(a, "n")
        c = lem.lemmatize(b, "r")
        d = lem.lemmatize(c, "a")
        bagW.append(d)
        
    final =[" ".join(bagW)]
    
    
    return final



