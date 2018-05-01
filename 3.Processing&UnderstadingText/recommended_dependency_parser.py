

sentence = 'The brown fox is quick and he is jumping over the lazy dog'

#Load dependencies
from spacy.lang.en import English
parser = English()
parsed_sent = parser(unicode(sentence))

print(parsed_sent) ,type(parsed_sent)

dependency_pattern = '{left}<---{word}[{w_type}]--->{right}\n--------'
for token in parsed_sent:
    print dependency_pattern.format(word=token.orth_,
                                  w_type=token.dep_,
                                  left=[t.orth_
                                            for t
                                            in token.lefts],
                                  right=[t.orth_
                                             for t
                                             in token.rights])

from nltk.parse.stanford import StanfordDependencyParser
sdp = StanfordDependencyParser(path_to_jar='/home/bhavani/work/Python/programs/NLTK/stanford-parser-full-2018-02-27/stanford-parser.jar',
                               path_to_models_jar='/home/bhavani/work/Python/programs/NLTK/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar')
result = list(sdp.raw_parse(sentence))
#print(result[0])
#print(type(result[0]))

dep_tree = [parse.tree() for parse in result][0]
print dep_tree
#dep_tree.draw()

# generation of annotated dependency tree shown in Figure 3-4
from graphviz import Source
dep_tree_dot_repr = [parse for parse in result][0].to_dot()
source = Source(dep_tree_dot_repr, filename="dep_tree", format="png")
source.view()

#Building our own dependecny parsers
import nltk
tokens = nltk.word_tokenize(sentence)

dependency_rules = """
'fox' -> 'The' | 'brown'
'quick' -> 'fox' | 'is' | 'and' | 'jumping'
'jumping' -> 'he' | 'is' | 'dog'
'dog' -> 'over' | 'the' | 'lazy'
"""

dependency_grammar = nltk.grammar.DependencyGrammar.fromstring(dependency_rules)
print dependency_grammar

dp = nltk.ProjectiveDependencyParser(dependency_grammar)
res = [item for item in dp.parse(tokens)]
tree = res[0]
print tree

tree.draw()