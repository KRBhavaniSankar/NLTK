"""
tokenizing the sentence by using delimiter like \n or ; or etc.
NLTK framework which provides various interfaces for performing sentence tokenization like


#sent_tokenize
#PunktSentenceTokenizer
#RegexpTokenizer
#Pre-trained sentence tokenization modles.
"""

import nltk
from nltk.corpus import gutenberg

from pprint import pprint
alice = gutenberg.raw(fileids='carroll-alice.txt')
print(type(alice))
sample_text = '''We will discuss briefly about the basic syntax, structure and
design philosophies. There is a defined hierarchical syntax for Python code
which you should remember when writing code! Python is a really powerful
programming language!'''

#Total characters in Alice in Wonderland
#print(len(alice))

#First 100 characters in the corpus
#print(alice[0:100])

default_st = nltk.sent_tokenize
alice_sentences = default_st(text = alice)
sample_sentences = default_st(text = sample_text)

"""
print('Total sentences in sample_text:', len(sample_sentences))
print('Sample text sentences :-')
pprint(sample_sentences)
print('\nTotal sentences in alice:', len(alice_sentences))
print('First 5 sentences in alice:-')
pprint(alice_sentences[0:5])
"""
#PunktSentenceTokenizer

punkt_st = nltk.tokenize.PunktSentenceTokenizer()
sample_sentences = punkt_st.tokenize(sample_text)

#pprint(sample_sentences)

SENTENCE_TOKENS_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s'
regex_st = nltk.tokenize.RegexpTokenizer(pattern=SENTENCE_TOKENS_PATTERN,gaps=True)
sample_sentences = regex_st.tokenize(sample_text)
pprint(len(sample_sentences))
