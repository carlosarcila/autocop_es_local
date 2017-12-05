#!/usr/bin/env python
# encoding: utf-8
#Este script descarga los tweets con un determinado contenido en el texto del mensaje usando API REST

import tweepy

#USAR LAS CLAVES QUE SAQUES EN https://apps.twitter.com
ckey=""
csecret=""
atoken=""
asecret=""

OAUTH_KEYS = {'consumer_key':ckey, 'consumer_secret':csecret,'access_token_key':atoken, 'access_token_secret':asecret}
auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
api = tweepy.API(auth)

#AQUI MODIFICA EL TEXTO DENTRO DEL TWEET QUE SE QUIERA BUSCAR Y LAS FECHAS
cricTweet = tweepy.Cursor(api.search, q='#Podemos', exclude= "retweets", lang="es", since="2017-12-03", until="2017-12-04").items()

#since="2017-07-01", until="2017-07-03"

for tweet in cricTweet:
    #print ((tweet.created_at, tweet.text.encode('utf-8')))
    #SI QUEREMOS QUE SE LEA MEJOR EN ESPANOL:
    #print tweet.created_at, tweet.text.encode('utf-8').decode("utf-8")
    #Y SIN FECHA:
    print ((tweet.text.encode('utf-8').decode("utf-8")))
