# -*- coding: utf-8 -*-
"""
Created on Fri APR 29 16:21 AM

@author: Bhavani
"""
#POS Taggers

import nltk

sentence = 'The brown fox is quick and he is jumping over the lazy dog'
tokens = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokens,tagset="universal")


print(tagged_sent)
print(type(tagged_sent))
print(type(tagged_sent[0]))

#Building your own POS Taggers
from nltk.corpus import treebank
data = treebank.tagged_sents()
train_data = data[:3500]
test_data =data[3500:]

# get a look at what each data point looks like
#print(train_data[0])

# remember tokens is obtained after tokenizing our sentence

tokens = nltk.word_tokenize(sentence)
print(tokens)

"""Default tagger, which inherits from the Sequential BackoffTagger base class and assigns the same userinput POS 
tag to each word
"""
from nltk.tag import DefaultTagger
dt = DefaultTagger("NN")
#Accuracy on test data
print(dt.evaluate(test_data))

print(dt.tag(tokens))

from nltk.tag import RegexpTagger
#define regex tag patterns

patterns = [
(r'.*ing$', 'VBG'),
(r'.*ed$', 'VBD'),
(r'.*es$', 'VBZ'),
(r'.*ould$', 'MD'),
(r'.*\'s$', 'NN$'),
(r'.*s$', 'NNS'),
(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
(r'.*', 'NN')]
rt = RegexpTagger(patterns)
# accuracy on test data
print(rt.evaluate(test_data))

from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger

ut = UnigramTagger(train_data)
bt = BigramTagger(train_data)
tt = TrigramTagger(train_data)

#testing perfomence of unigram tagger
print(ut.evaluate(test_data))
print(ut.tag(tokens))

#testing perfomence of bigram tagger
print(bt.evaluate(test_data))
print(bt.tag(tokens))

#testing perfomence of trigram tagger
print(tt.evaluate(test_data))
print(tt.tag(tokens))

def combined_tagger(train_data,taggers,backoff =None):
    for tagger in taggers:
        backoff = tagger(train_data,backoff=backoff)
    return backoff

ct = combined_tagger(train_data=train_data,taggers=[UnigramTagger,BigramTagger,TrigramTagger],backoff=rt)
print(ct.evaluate(test_data))

print(ct.tag(tokens))