"""
word_tokenize
TreebankWordTokenizer
RegexpTokenizer
Inheried tokenizers from RegexpTokenizer
"""
import nltk
from nltk import word_tokenize

default_wt = nltk.word_tokenize

sentence = "The brown fox wasn't that quick and he couldn't win the race"

words = default_wt(sentence)
print(words)
#print(type(words))

treebank_wt = nltk.TreebankWordTokenizer()
words = treebank_wt.tokenize(sentence)
print(words)

#Pattern to identify tokens themselves
token_pattern =r"\w+"
regex_wt = nltk.RegexpTokenizer(pattern = token_pattern,gaps=False)
words = regex_wt.tokenize(sentence)
print(words)

#Pattern to identiy gaps in tokens
GAP_PATTERN = r"\s+"
regex_wt = nltk.RegexpTokenizer(pattern=GAP_PATTERN,gaps=True)
words =regex_wt.tokenize(sentence)
print(words)
# get start and end indices of each token and then print them

word_indices = list(regex_wt.span_tokenize(sentence))
print(word_indices)
print([sentence[start:end] for start, end in word_indices])


wordpunkt_wt = nltk.WordPunctTokenizer()
words = wordpunkt_wt.tokenize(sentence)
print(words)