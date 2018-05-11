# -*- coding: utf-8 -*-
"""
Created on Wed May 09 18:42:26 2018

@author: Bhavani
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split

def get_data():
    data = fetch_20newsgroups(subset='all',
                              shuffle=True,
                              remove=('headers','footers','quotes'))
    return data


def remove_empty_docs(corpus,labels):
    filtered_corpus = []
    filtered_labels = []
    for doc,label in zip(corpus,labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)
    return filtered_corpus,filtered_labels


def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels,
                                                        test_size=0.33, random_state=42)
    return train_X, test_X, train_Y, test_Y



dataset = get_data()

#print(dataset)
#print(type(dataset))

corpus, labels = dataset.data, dataset.target
corpus,labels = remove_empty_docs(corpus,labels)

#print('Sample document:', corpus[10])
#print('Class label:',labels[10])
#print('Actual class label:', dataset.target_names[labels[10]])

train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,
                                                                        labels,
                                                                        test_data_proportion=0.3)

