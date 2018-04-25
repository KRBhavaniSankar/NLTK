#load the wordnet corpus

from nltk.corpus import wordnet as wn

word = 'generic'   #Taking hike as our word of interest

#get word synsets

word_synsets = wn.synsets(word)

print(word_synsets)

#get details of each synonim in synset

for synset in word_synsets:
    print("Synset Name :",synset.name())
    print("POS Tag :",synset.pos())
    print("Definition :",synset.definition())
    print("Examples :",synset.examples())
