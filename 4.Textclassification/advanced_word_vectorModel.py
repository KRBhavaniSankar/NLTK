# -*- coding: utf-8 -*-
"""
Created on Mon May 07 14:55:10 2018

@author: Bhavani
"""

from sklearn.feature_extraction.text import CountVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


from sklearn.feature_extraction.text import TfidfTransformer


def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


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


def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)


def tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model, num_features):
    word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)]
                   if tfidf_vocabulary.get(word)
                   else 0 for word in words]
    word_tfidf_map = {word: tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}

    feature_vector = np.zeros((num_features,), dtype="float64")
    vocabulary = set(model.wv.index2word)
    wts = 0.
    for word in words:
        if word in vocabulary:
            word_vector = model[word]
            weighted_word_vector = word_tfidf_map[word] * word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vector)
    if wts:
        feature_vector = np.divide(feature_vector, wts)

    return feature_vector


def tfidf_weighted_averaged_word_vectorizer(corpus, tfidf_vectors,
                                            tfidf_vocabulary, model, num_features):
    docs_tfidfs = [(doc, doc_tfidf)
                   for doc, doc_tfidf
                   in zip(corpus, tfidf_vectors)]
    features = [tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,
                                           model, num_features)
                for tokenized_sentence, tfidf in docs_tfidfs]
    return np.array(features)


CORPUS = [
    'the sky is blue',
    'sky is blue and sky is beautiful',
    'the beautiful sky is so blue',
    'i love blue cheese'
]

new_doc = ['loving this blue sky today']

import pandas as pd


def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                      columns=feature_names)
    print(df)



bow_vectorizer, bow_features = bow_extractor(CORPUS)
features = bow_features.todense()
print(features)

new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features = new_doc_features.todense()
print(new_doc_features)

feature_names = bow_vectorizer.get_feature_names()
print(feature_names)

display_features(features, feature_names)
display_features(new_doc_features, feature_names)

import numpy as np

feature_names = bow_vectorizer.get_feature_names()

tfidf_trans, tdidf_features = tfidf_transformer(bow_features)
features = np.round(tdidf_features.todense(), 2)
display_features(features, feature_names)

nd_tfidf = tfidf_trans.transform(new_doc_features)
nd_features = np.round(nd_tfidf.todense(), 2)
display_features(nd_features, feature_names)

import scipy.sparse as sp
from numpy.linalg import norm

feature_names = bow_vectorizer.get_feature_names()

# compute term frequency
tf = bow_features.todense()
tf = np.array(tf, dtype='float64')

# show term frequencies
display_features(tf, feature_names)

# build the document frequency matrix
df = np.diff(sp.csc_matrix(bow_features, copy=True).indptr)
df = 1 + df  # to smoothen idf later

# show document frequencies
display_features([df], feature_names)

# compute inverse document frequencies
total_docs = 1 + len(CORPUS)
idf = 1.0 + np.log(float(total_docs) / df)

# show inverse document frequencies
display_features([np.round(idf, 2)], feature_names)

# compute idf diagonal matrix
total_features = bow_features.shape[1]
idf_diag = sp.spdiags(idf, diags=0, m=total_features, n=total_features)
idf = idf_diag.todense()

# print the idf diagonal matrix
print(np.round(idf, 2))

# compute tfidf feature matrix
tfidf = tf * idf

# show tfidf feature matrix
display_features(np.round(tfidf, 2), feature_names)

# compute L2 norms
norms = norm(tfidf, axis=1)

# print norms for each document
print(np.round(norms, 2))

# compute normalized tfidf
norm_tfidf = tfidf / norms[:, None]

# show final tfidf feature matrix
display_features(np.round(norm_tfidf, 2), feature_names)

# compute new doc term freqs from bow freqs
nd_tf = new_doc_features
nd_tf = np.array(nd_tf, dtype='float64')

# compute tfidf using idf matrix from train corpus
nd_tfidf = nd_tf * idf
nd_norms = norm(nd_tfidf, axis=1)
norm_nd_tfidf = nd_tfidf / nd_norms[:, None]

# show new_doc tfidf feature vector
display_features(np.round(norm_nd_tfidf, 2), feature_names)


tfidf_vectorizer, tdidf_features = tfidf_extractor(CORPUS)
display_features(np.round(tdidf_features.todense(), 2), feature_names)

nd_tfidf = tfidf_vectorizer.transform(new_doc)
display_features(np.round(nd_tfidf.todense(), 2), feature_names)

import gensim
import nltk

TOKENIZED_CORPUS = [nltk.word_tokenize(sentence)
                    for sentence in CORPUS]
tokenized_new_doc = [nltk.word_tokenize(sentence)
                     for sentence in new_doc]

model = gensim.models.Word2Vec(TOKENIZED_CORPUS,
                               size=10,
                               window=10,
                               min_count=2,
                               sample=1e-3)


avg_word_vec_features = averaged_word_vectorizer(corpus=TOKENIZED_CORPUS,
                                                 model=model,
                                                 num_features=10)
print(np.round(avg_word_vec_features, 3))

nd_avg_word_vec_features = averaged_word_vectorizer(corpus=tokenized_new_doc,
                                                    model=model,
                                                    num_features=10)
print(np.round(nd_avg_word_vec_features, 3))


corpus_tfidf = tdidf_features
vocab = tfidf_vectorizer.vocabulary_
wt_tfidf_word_vec_features = tfidf_weighted_averaged_word_vectorizer(corpus=TOKENIZED_CORPUS,
                                                                     tfidf_vectors=corpus_tfidf,
                                                                     tfidf_vocabulary=vocab,
                                                                     model=model,
                                                                     num_features=10)
print(np.round(wt_tfidf_word_vec_features, 3))

nd_wt_tfidf_word_vec_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_new_doc,
                                                                        tfidf_vectors=nd_tfidf,
                                                                        tfidf_vocabulary=vocab,
                                                                        model=model,
                                                                        num_features=10)
print(np.round(nd_wt_tfidf_word_vec_features, 3))
