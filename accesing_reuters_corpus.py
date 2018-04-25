#load the reurters corpus

from nltk.corpus import reuters

#Reuters corpus having 10,788 reuters news documents from around 90 different categories

#print the total categories

print('Total categories :',len(reuters.categories()))

#print categoires

print(reuters.categories())

#get sentences in housing and income categories
sentences = reuters.sents(categories =['housing','income'])
sentences = [' '.join(sentence_token) for sentence_token in sentences]
print(sentences[0:5])

#Field based acccess
print(reuters.fileids(categories =["housing","income"]))

print(reuters.sents(fileids = ['test/16118', 'test/18672']))