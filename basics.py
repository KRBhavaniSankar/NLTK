from nltk.corpus import brown



#printing categories in brown corpus

print( brown.categories)

#tokenized sentences
print(brown.sents(categories = "mystery"))

#POS taggged sentences

print(brown.tagged_sents(categories ="mystery"))

#get sentences in natural form

sentences = brown.sents(categories = "mystery")
sentences = [" ".join(sentence_token ) for sentence_token in sentences]
print(sentences)