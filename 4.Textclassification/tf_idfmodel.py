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
print(new_doc_features)

feature_names = bow_vectorizer.get_feature_names()
print(feature_names)

def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                      columns=feature_names)
    print(df)



display_features(features, feature_names)
display_features(new_doc_features, feature_names)

feature_names = bow_vectorizer.get_feature_names()

tfidf_trans, tdidf_features = tfidf_transformer(bow_features)
features = np.round(tdidf_features.todense(), 2)
display_features(features, feature_names)

nd_tfidf = tfidf_trans.transform(new_doc_features)
nd_features = np.round(nd_tfidf.todense(), 2)
display_features(nd_features, feature_names)
