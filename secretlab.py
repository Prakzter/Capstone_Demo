import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
import re
import time, datetime, calendar

import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import KeyedVectors
from gensim.models import CoherenceModel
from gensim.models import Word2Vec, Doc2Vec
from pprint import pprint

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import utils


import pickle
import textcleaner



### code for frontend ###
st.image('logo-1.png', use_column_width=True)
st.title("Brand Insights For Secretlab")
st.subheader("This app aims to serve as a simple frontend to demonstrate our text classification model.")


# st.write("---"*20)
# '''#### Don't have any text to try out? Copy our sample text here and analyse below...
#    ####  
# '''


# if st.checkbox('Sample text for Stocks Investment'):
#     st.write(""" I am currently using StashAway as my main investment. I wish to try stocks/ETF index funds, any suggestions on where I should invest?""")

# if st.checkbox('Sample text for Real Estate Investment'):
#     st.write(""" At a young age (e.g 23 years old) with Low income (<$4000), is it advisable to venture 
#     commercial and industrial property investment instead, and use the profits to fund residential property?""")

# if st.checkbox('Sample text for mixed class'):
#     st.write('''There's a huge undeniable bubble going on in the stock market, it's just a question how about long further it will stretch till it burst.
# But I'm in a dilemma, I found a nice coming apartment that will be completed in 6-7 months. It's slightly overpriced, but not too bad. How ever, is it a terrible idea to buy an apartment right now?
# Not sure what market crash will result. Stocks will decline alot, but how about real estate?
# It's also said that pricing on housing/apartment is all time high in my country.''')
# st.write("---"*20)

stats = pd.read_csv('vidstats.csv')

filter_month = st.slider('month', 1, 12, 5)
filter_year = st.slider('year', 2016, 2020, 2018)
filter_likeCount = st.slider('# of likes', 50, 8443, 1500)

st.dataframe(stats)


author_stats = stats.groupby(['video_author'])['viewCount','commentCount','likeCount','dislikeCount'].sum()
author_stats.sort_values(by=['viewCount'],inplace=True,ascending=False)

@st.cache
new_model = Word2Vec.load('mywordvecs.model')
d2v_model = Doc2Vec.load('dm_model.model')

def predict(text_list):
    model = pickle.load(open('clf.pkl','rb'))
    return model.predict(text_list)


def predict_proba(text_list):
    model = pickle.load(open('clf.pkl','rb'))
    return model.predict_proba(text_list)




def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 50), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 800 to 20 dimensions with PCA
    reduc = PCA(n_components=11).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=42, perplexity=10, learning_rate=150).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title(f't-SNE visualization for {word.title()}')

st.write("---"*20)

st.header("Visualisation of trigger words")
trigger_word = st.text_input('', 'Input your keyword here (in lower caps pls)...')
try:
    st.pyplot(tsnescatterplot(new_model, trigger_word, []))
except:
    st.warning("OOps, looks like that word hasn't been mentioned in the comments. Please try another word")

st.write("---"*20)




st.subheader('Enter your Youtube comment here:')
message = st.text_area('',"""input here...""")

# code to transform msg
topics = ['Firmness of the seat causing back issues', 
            'Durability of the chair',
            'Considering between Fabric and Leather',
            'Raving about the Omega chair',
            'Lumbar support',
            'Compliments to the actual video'
        ]

# cleaned_text = textcleaner.transform_text(message)
if st.checkbox('Show cleaned text (removing non-word characters,digits and lemmatise text)'):
    cleaned_up = [word for word in simple_preprocess(message)]
    st.text(cleaned_up)

    
vector = d2v_model.infer_vector(cleaned_up)


st.subheader(" Input comment transformed into a feature vector by Doc2Vec")
st.write(vector)



if st.button('Analyse post'):
    confidence = predict_proba([vector])[0]

    st.subheader("The level of confidence betwen the topics:")
    st.write("---"*20)
    for i,value in enumerate(confidence):
        st.write(topics[i],round(value,4))
    
    
    st.write("---"*20)
    prediction = predict([vector])[0]
    st.subheader('\n\nGreat this comment should belong to the topic of:')
    st.text(topics[prediction])
    st.balloons()

    
