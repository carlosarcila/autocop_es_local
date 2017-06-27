#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Lexicon
#https://github.com/dlizcano/tuit_sentimiento/blob/master/data/lexicon/subjectivity.csv#

import math
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# AFINN-111 is as of June 2011 the most recent version of AFINN
# Lexicón en español:
# filenameAFINN = 'AFINN/AFINN-111.txt'
filenameAFINN = "DICT/lexicon_es.txt"

afinn = dict(map(lambda (w, s): (w, int(s)), [ws.strip().split('\t') for ws in open(filenameAFINN) ]))

# Word splitter pattern
pattern_split = re.compile(r"\W+")

def sentiment(text):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence.
    """
    words = pattern_split.split(text.lower())
    sentiments = map(lambda word: afinn.get(word, 0), words)
    if sentiments:
        # How should you weight the individual word sentiments?
        # You could do N, sqrt(N) or 1 for example. Here I use sqrt(N)
        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))

    else:
        sentiment = 0
    return sentiment



if __name__ == '__main__':
    # Single sentence example:
    text = "Buenos días. Hace una mañana bonita y agradable. esta es una prueba de Python en Español. Bienvenidos todos a NLTK. Seguro que os va a gusta"
    print("%6.2f %s" % (sentiment(text), text))

    # No negation and booster words handled in this approach
    text = "El día de hoy ha sido muy malo.. No he entendido nada de Python y no me gusta el calor porque considero que es muy desagradable"
    print("%6.2f %s" % (sentiment(text), text))