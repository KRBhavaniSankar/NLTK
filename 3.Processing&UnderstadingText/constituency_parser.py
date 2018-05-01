# -*- coding: utf-8 -*-
"""
Created on Tue May 01 15:09:49 2018

@author: Bhavani
"""

sentence = 'The brown fox is quick and he is jumping over the lazy dog'

from nltk.parse.stanford import StanfordParser

scp = StanfordParser(path_to_jar='/home/bhavani/work/Python/programs/NLTK/stanford-parser-full-2018-02-27/stanford-parser.jar',
path_to_models_jar='/home/bhavani/work/Python/programs/NLTK/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar')


result = list(scp.raw_parse(sentence))
#print result[0]
#result[0].draw()

#Building our own constituency parsers

import nltk
from nltk.grammar import Nonterminal
from nltk.corpus import treebank

training_set = treebank.parsed_sents()

print training_set[1]

# extract the productions for all annotated training sentences
treebank_productions = list(
                        set(production
                            for sent in training_set
                            for production in sent.productions()
                        )
                    )

print treebank_productions[0:10]


# add productions for each word, POS tag
for word, tag in treebank.tagged_words():
	t = nltk.Tree.fromstring("("+ tag + " " + word  +")")
	for production in t.productions():
		treebank_productions.append(production)

# build the PCFG based grammar
treebank_grammar = nltk.grammar.induce_pcfg(Nonterminal('S'),
                                         treebank_productions)

# build the parser
viterbi_parser = nltk.ViterbiParser(treebank_grammar)

# get sample sentence tokens
tokens = nltk.word_tokenize(sentence)

# get parse tree for sample sentence
result = list(viterbi_parser.parse(tokens))


# get tokens and their POS tags
from pattern.en import tag as pos_tagger
tagged_sent = pos_tagger(sentence)

print tagged_sent

# extend productions for sample sentence tokens
for word, tag in tagged_sent:
    t = nltk.Tree.fromstring("("+ tag + " " + word  +")")
    for production in t.productions():
		treebank_productions.append(production)

# rebuild grammar
treebank_grammar = nltk.grammar.induce_pcfg(Nonterminal('S'),
                                         treebank_productions)

# rebuild parser
viterbi_parser = nltk.ViterbiParser(treebank_grammar)

# get parse tree for sample sentence
result = list(viterbi_parser.parse(tokens))

print result[0]
result[0].draw()                  