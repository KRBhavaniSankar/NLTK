# -*- coding: utf-8 -*-
"""
Created on Fri May 04 22:59:10 2018

@author: Bhavani
"""

from sklearn.feature_extraction.text import CountVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]
new_doc = ['loving this blue sky today']
from sklearn.feature_extraction.text import TfidfTransformer

def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer,tfidf_matrix


from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features



import numpy as np
import pandas as pd
bow_vectorizer, bow_features = bow_extractor(CORPUS)
features = bow_features.todense()
#print(features)


new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features = new_doc_features.todense()
#print(new_doc_features)

feature_names = bow_vectorizer.get_feature_names()
#print(feature_names)

def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                      columns=feature_names)
    print(df)



#display_features(features, feature_names)
#display_features(new_doc_features, feature_names)

feature_names = bow_vectorizer.get_feature_names()

tfidf_trans, tdidf_features = tfidf_transformer(bow_features)
features = np.round(tdidf_features.todense(), 2)
#display_features(features, feature_names)

nd_tfidf = tfidf_trans.transform(new_doc_features)
nd_features = np.round(nd_tfidf.todense(), 2)
#display_features(nd_features, feature_names)

import scipy.sparse as sp
from numpy.linalg import norm
feature_names = bow_vectorizer.get_feature_names()

#compute term frequency

tf = bow_features.todense()
tf = np.array(tf,dtype='float64')

#show term frequencies
display_features(tf,feature_names)

# build the document frequency matrix
df = np.diff(sp.csc_matrix(bow_features, copy=True).indptr)
df = 1 + df # to smoothen idf later

display_features([df],feature_names)

total_docs = 1+ len(CORPUS)
idf = 1.0+ np.log(float(total_docs)/df)
#show inverse document frequencies
display_features([np.round(idf,2)],feature_names)

#compute the idf diagonal matrix
print(np.round(idf,2))

tf_idf = tf * idf

display_features(np.round(tf_idf,2),feature_names)

#compute l2 norms
norms = norm(tf_idf,axis=1)
#print norms for each document
print(np.round(norms,2))

# compute normalized tfidf
norm_tfidf = tf_idf / norms[:, None]
# show final tfidf feature matrix
#display_features(np.round(norm_tfidf, 2), feature_names)

#compute the new doc term freqs from bow freqs
nd_tf = new_doc_features
nd_tf = np.array(nd_tf,dtype="float64")

# compute tfidf using idf matrix from train corpus
nd_tfidf = nd_tf*idf
nd_norms = norm(nd_tfidf, axis=1)
norm_nd_tfidf = nd_tfidf / nd_norms[:, None]
# show new_doc tfidf feature vector
#display_features(np.round(norm_nd_tfidf, 2), feature_names)

#Generic function that can directly compute the tfidf-based feature vectors

from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_extractor(corpus,ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer,features

tfidf_vectorizer, tdidf_features = tfidf_extractor(CORPUS)
#display_features(np.round(tdidf_features.todense(), 2), feature_names)

nd_tfidf = tfidf_vectorizer.transform(new_doc)
display_features(np.round(nd_tfidf.todense(), 2), feature_names)