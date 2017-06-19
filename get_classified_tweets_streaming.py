#!/usr/bin/python
# -*- coding: UTF-8 -*-
#ESTE SCRIPT DESCARGA LOS TWEETS DEL STREAMING DE TWITTER CON UNA PALABRA CLAVE
#PINTA POR PANTALLA EL MENSAJE CON SU SENTIMIENTO (POS o NEG) Y EL NIVEL DE CONFIANZA DE ESE SENTIMIENTO
#ESCRIBE EN ARCHIVO APARTE LOS SENTIMIENTOS DE LOS MENSAJES QUE HAN ALCANZADO UNA CONFIANZA DE POR LO MENOS .80

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey=""
csecret=""
atoken=""
asecret=""



class listener(StreamListener):

    def on_data(self, data):

		all_data = json.loads(data)

		tweet = all_data["text"]
		sentiment_value, confidence = s.sentiment(tweet)
		print(tweet, sentiment_value, confidence)

		if confidence*100 >= 80:
			output = open("twitter-out.txt","a")
			output.write(sentiment_value)
			output.write('\n')
			output.close()

		return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["#Podemos"])
