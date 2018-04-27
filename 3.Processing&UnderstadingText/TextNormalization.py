
"""
#Text Normalization : It is the process that consists of a series of steps that shoudl be followed to wrangle, clean and
standerddize texual data into a from that could be consumed by other NLP and analysis systms and applications.

various things involved ehre like cleaning,case conversion, correct speelings,removing stop words and other unnecessary
terms stemming and lemmetizaton.  It is also called text cleansing or wrangling.

"""

import nltk
import re
import string
from pprint import pprint

import nltk
import re
import string
from pprint import pprint

corpus = ["The brown fox wasn't that quick and he couldn't win the race",
          "Hey that's a great deal! I just bought a phone for $199",
          "@@You'll (learn) a **lot** in the book. Python is an amazing language!@@"]


def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return word_tokens


token_list = [tokenize_text(text)
              for text in corpus]
pprint(token_list)



def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens


filtered_list_1 = [filter(None, [remove_characters_after_tokenization(tokens)
                                 for tokens in sentence_tokens])
                   for sentence_tokens in token_list]
print(filtered_list_1)



def remove_characters_before_tokenization(sentence,
                                          keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]'
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    else:
        PATTERN = r'[^a-zA-Z0-9 ]'
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence


filtered_list_2 = [remove_characters_before_tokenization(sentence)
                   for sentence in corpus]
print(filtered_list_2)


cleaned_corpus = [remove_characters_before_tokenization(sentence, keep_apostrophes=True)
                  for sentence in corpus]
print(cleaned_corpus)


