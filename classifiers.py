#!/usr/bin/python
# -*- coding: UTF-8 -*-
#ESTE SCRIPT ME PERMITE AHORRAR TIEMPO, GUARDANDO EN FORMATO PICKLE LAS CARACTERISTICAS Y LOS CLASIFICADORES
#ESTOS DOCUMENTOS ENTRENADOS VAN A UN DIRECTORIO APARTE LLAMADO 'pickled_algos'
#PARA PODER SER USADOS DESDE EL MODULO DE ANALISIS DE SENTIMIENTOS

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

#adaptacion SPANISH
from nltk.corpus import cess_esp
nltk.tag.mapping._load_universal_map("es-cast3lb") 
mapdict = nltk.tag.mapping._MAPPINGS["es-cast3lb"]["universal"] 
alltags = set(t for w, t in cess_esp.tagged_words())
for tag in alltags:
    if len(tag) <= 2:   # These are complete
        continue
    mapdict[tag] = mapdict[tag[:2]]

cess_esp._tagset = "es-cast3lb"
from nltk import UnigramTagger as ut
from nltk import BigramTagger as bt
cess_sents = cess_esp.tagged_sents(tagset='universal')
uni_tag = ut(cess_sents, backoff=nltk.DefaultTagger('X'))




class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()

#He tenido que introducir este par de lineas en mi mac porque estaba dando problemas para reconocer el tipo de formato
short_pos = short_pos.decode('utf-16', 'ignore')
short_neg = short_neg.decode('utf-16,', 'ignore')


# move this up here
all_words = []
documents = []


#Tipos de palabras que usa el clasificador
#ADJ: adjective, ADP: adposition, ADV: adverb, AUX: auxiliary, CCONJ: coordinating conjunction, DET: determiner
#INTJ: interjection, NOUN: noun, NUM: numeral, PART: particle, PRON: pronoun, PROPN: proper noun
#PUNCT: punctuation, SCONJ: subordinating conjunction, SYM: symbol, VERB: verb, X: other
#Incluir los tipos que queramos en la lista: allowed_word_types = ["NOUN", "ADJ", "VERB"]

allowed_word_types = ['NOUN', 'ADJ', 'VERB']

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = nltk.word_tokenize(p)
    pos = uni_tag.tag(words)
    for w in pos:
        if w[1] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = nltk.word_tokenize(p)
    pos = uni_tag.tag(words)
    for w in pos:
        if w[1] in allowed_word_types:
            all_words.append(w[0].lower())


#He abierto manualmente un directorio llamado "pickled_algos"
###OJO aqui se cambia estructura del PICKLE
pickle.dump( documents, open( "pickled_algos/documents.pickle", "wb" ) )

#save_documents = open("pickled_algos/documents.pickle","wb")
#pickle.dump(documents, save_documents)
#save_documents.close()


all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]

#
pickle.dump( word_features, open( "pickled_algos/word_features5k.pickle", "wb" ) )
#save_word_features = open("pickled_algos/word_features5k.pickle","wb")
#pickle.dump(word_features, save_word_features)
#save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print(len(featuresets))


#NUMERO DE INSTANCIAS, CASOS O EJEMPLOS PARA ENTRENAR O PROBAR
#EL EJEMPLO ES DE 10000
training_set = featuresets[:7000]
testing_set = featuresets[7000:]

#he tenido que agregar esto que no estaba
pickle.dump( featuresets, open( "pickled_algos/featuresets.pickle", "wb" ) )


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

###############

#
pickle.dump( classifier, open( "pickled_algos/originalnaivebayes5k.pickle", "wb" ) )
#save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


#
pickle.dump( MNB_classifier, open( "pickled_algos/MNB_classifier5k.pickle", "wb" ) )
#save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
#pickle.dump(MNB_classifier, save_classifier)
#save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)


#
pickle.dump( BernoulliNB_classifier, open( "pickled_algos/BernoulliNB_classifier5k.pickle", "wb" ) )
#save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
#pickle.dump(BernoulliNB_classifier, save_classifier)
#save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

#
pickle.dump( LogisticRegression_classifier, open( "pickled_algos/LogisticRegression_classifier5k.pickle", "wb" ) )
#save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
#pickle.dump(LogisticRegression_classifier, save_classifier)
#save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

#
pickle.dump( LinearSVC_classifier, open( "pickled_algos/LinearSVC_classifier5k.pickle", "wb" ) )
#save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
#pickle.dump(LinearSVC_classifier, save_classifier)
#save_classifier.close()


##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

#
pickle.dump( SGDC_classifier, open( "pickled_algos/SGDC_classifier5k.pickle", "wb" ) )
#save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
#pickle.dump(SGDC_classifier, save_classifier)
#save_classifier.close()
