# -*- coding: utf-8 -*-
"""
Created on Fri May 04 16:08:10 2016

@author: Bhavani
"""
from sklearn.feature_extraction.text import CountVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    print( vectorizer.vocabulary_)
    #print(vectorizer.vocabulary)
    #print(vectorizer.fixed_vocabulary_)
    #print(features)
    return vectorizer, features

CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]
new_doc = ['loving this blue sky today']


bow_vectorizer, bow_features = bow_extractor(CORPUS)
features = bow_features.todense()
print(features)