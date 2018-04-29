# -*- coding: utf-8 -*-
"""
Created on Fri APR 29 10:04 AM

@author: Bhavani
"""

"""
The NLTK package has several implementation for stemmers, These stemmers are implemented in the stem module , which inherits the stemmer interface in the nltk.stem.api module.
One of the most popular stemmers is the PorterStemmer.  There also exists poter2 algorithm which is imporvements of original stemmig algorithm.
"""

#Porterstemmer
from nltk.stem import PorterStemmer
ps = PorterStemmer()

words_list = ["jumping","jumps","jumped","jump"]
for w in words_list:
    print(ps.stem(w))

#print(ps.stem("lying"))
#print(ps.stem("strange"))

from nltk.stem import LancasterStemmer
ls = LancasterStemmer()
for w in words_list:
    print(ls.stem(w))

print(ls.stem("lying"))
print(ls.stem("strange"))

"""
There are several other stemmers, including RegexpStemmer , where you can build
your own stemmer based on user-defined rules , and SnowballStemmer , which supports
stemming in 13 different languages besides English.
"""

#Regex Based stemmer
from nltk.stem import RegexpStemmer
rs = RegexpStemmer("ing$|s$|ed$",min=4)


for w in words_list:
    print(rs.stem(w))

print(rs.stem("lying"))
print(rs.stem("strange"))

#Snow Ball stemmer
from nltk.stem import SnowballStemmer
ss = SnowballStemmer("german")


print("supported languages are :",SnowballStemmer.languages)

german_cars = "autobahnen"
print(ss.stem(german_cars))