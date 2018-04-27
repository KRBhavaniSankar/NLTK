# -*- coding: utf-8 -*-
"""
Created on Fri APR 27 15:54

@author: Bhavani
"""
import re,collections
import io
def tokens(text):
    """
    Get all words from the corpus
    :param text:
    :return:
    """
    return re.findall('[a-z]+',text.lower())

#text = 'Hi this is bhavani'
#print(text.lower())
#print(text.upper())

WORDS = tokens(open('/home/bhavani/nltk_data/big-text.txt','r').read())
#print(WORDS)
word_counts = collections.Counter(WORDS)
#print(word_counts.most_common(10))

def edits0(word):
    """
    Return all strings that are zero edits away
    from the input word (i.e., the word itself).
    """
    return {word}

def edits1(word):
    """
    Return all strings that are one edit away
    from the input word.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    def splits(word):
        """
        Return a list of all possible (first, rest) pairs
        that the input word is made of.
        """
        return [(word[:i], word[i:]) for i in range(len(word)+1)]
    pairs = splits(word)
    deletes = [a+b[1:] for (a,b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces =[a+c+b[1:] for (a,b) in pairs for c in alphabet if b]
    inserts = [a+c+b for (a,b) in pairs for c in alphabet]
    return set(deletes+transposes+replaces+inserts)

def edits2(word):
    """Returns all the string that are two edits away from the input word"""
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def known(words):
    """
    :param words: words
    :return:Return the subset of words that are actually
    in our WORD_COUNTS dictionary.
    """
    return {w for w in words if w in word_counts}

word = 'fynaly'

#print(known(edits0(word)))
#print(edits1(word))
#print(known(edits2(word)))

#candidates = (known(edits0(word)) or known(edits1(word)) or known(edits2(word)) or [word])
#print(candidates)

def correct(word):
    """
    # Priority is for edit distance 0, then 1, then 2
    # else defaults to the input word itself.
    :param word: texual word
    :return:Get the best correct spelling for the input word
    """
    candidates = (known(edits0(word)) or known(edits1(word)) or known(edits2(word)) or [word])
    return max(candidates,key=word_counts.get)

parse_string = word.lower()
correct_word = correct(parse_string)
print("Given Word is:",word)
print("Correct Word is:",correct_word)

'''
def correct_match(match):
    """

    :param match: spell-correct word in match
    :return: preserve proper upper/lower/title case
    """
    word =match.group()
    def case_of(text):
        """

        :param text:
        :return: Case-function apporpriate for text :upper,lower,title,or jst str.:
        """
        return (str.upper if text.isupper() else str.lower if text.islower() else str.title if text.istitle() else str)
    return case_of(word)(correct(word.lower()))

def correct_text_generic(text):
    """

    :param text:
    :return: correct all the words within a text, returning the corrected text.
    """
    return re.sub("[a-zA-Z]+",correct_match(match),text)

res = correct_text_generic(word)

print(res)
'''

