#!/usr/bin/env python
# encoding: utf-8
#Este Script se conecta al Streaming de Twitter y descarga los tweets PARSEADOS SIN RT Y LOS FULL TWEETS EN CASO DE ESTAR TRUNCADOS
#en formato Json que contengan en el texto una palabra -ejemplo "#Podemos"
#IMPRIME solo las partes del tweet que le pidamos -p.e. "text", "user",,


#Correr desde consola con Python2 y muestra resultados por la misma consola
#python2 get_full_tweets_from_streaming.py
#para escribir los resultados en un archivo:
#python2 get_full_tweets_from_streaming.py > twitter2.txt


from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import time

#consumer key, consumer secret, access token, access secret.
ckey=""
csecret=""
atoken=""
asecret=""

class listener(StreamListener):
    def on_status(self, status):
        if not status.retweeted and 'RT @' not in status.text:
            try:
                tweet = status.extended_tweet['full_text']
            except AttributeError:
                tweet = status.text

            print tweet
        else:
            return

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Ciudadanos"])


