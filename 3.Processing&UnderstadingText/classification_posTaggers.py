# -*- coding: utf-8 -*-
"""
Created on Fri APR 29 17:10 PM

@author: Bhavani
"""
#POS Taggers classification

from nltk.classify import NaiveBayesClassifier
from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.corpus import treebank
import nltk

sentence = 'The brown fox is quick and he is jumping over the lazy dog'
tokens = nltk.word_tokenize(sentence)
data = treebank.tagged_sents()
train_data = data[:3500]
test_data =data[3500:]

nbt = ClassifierBasedPOSTagger(train=train_data,
                               classifier_builder= NaiveBayesClassifier.train)

#evaluate tagger on the test data and sample sentence

print(nbt.evaluate(test_data))
print(nbt.tag(tokens))
