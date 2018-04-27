# -*- coding: utf-8 -*-
"""
Created on Fri APR 27 15:06

@author: Bhavani
"""
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import re
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
words = word_tokenize(data)
#print(words)

stopWords =set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered =[]

for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

#print(wordsFiltered)

#Removing Repeated characters
sample_sentene = 'My school is so so soooooo reallllyyyyyy amaaaazingggggg'
sample_sentene_tokens = word_tokenize(sample_sentene)
print(sample_sentene_tokens)

from nltk.corpus import wordnet


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


print(remove_repeated_characters(sample_sentene_tokens))