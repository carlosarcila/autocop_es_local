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
cricTweet = tweepy.Cursor(api.search, q='#barcelona', exclude= "retweets", lang="es", tweet_mode='extended').items()

for tweet in cricTweet:
    #print tweet.created_at, tweet.full_text
    #Y SIN FECHA:
    print tweet.full_text
