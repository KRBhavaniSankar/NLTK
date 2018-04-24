from nltk.corpus import brown



#printing categories in brown corpus

print( brown.categories)

#tokenized sentences
#print(brown.sents(categories = "mystery"))

#POS taggged sentences

#print(brown.tagged_sents(categories ="mystery"))

#get sentences in natural form

sentences = brown.sents(categories = "mystery")
sentences = [" ".join(sentence_token ) for sentence_token in sentences]
#print(sentences)

#get tagged words
tagged_words = brown.tagged_words(categories ="mystery")
print(type(tagged_words))
print(tagged_words)

# get nouns from tagged words
nouns = [(word, tag) for word, tag in tagged_words if any(noun_tag in tag for noun_tag in ['NP', 'NN'])]

print(nouns[0:10])