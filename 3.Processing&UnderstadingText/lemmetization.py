# -*- coding: utf-8 -*-
"""
Created on Fri APR 29 10:49 AM

@author: Bhavani
"""


from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

#Lemmetization nouns
print(wnl.lemmatize("cars","n"))
print(wnl.lemmatize("men","n"))

#Lemmetization verb
print(wnl.lemmatize("running","v"))
print(wnl.lemmatize("ate","v"))

#Lemmetization adjectives
print(wnl.lemmatize("saddest","a"))
print(wnl.lemmatize("fancier","a"))