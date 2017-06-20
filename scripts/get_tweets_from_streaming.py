#Este Script se conecta al Streaming de Twitter y descarga los tweets completos
#en formato Json que contengan en el texto una palabra -ejemplo "#BigData"

#Correr desde consola con Python2 y muestra resultados por la misma consola
#python2 twitter.py
#para escribir los resultados en un archivo:
#python2 twitter.py > twitter.txt



from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


#consumer key, consumer secret, access token, access secret.
ckey=""
csecret=""
atoken=""
asecret=""

class listener(StreamListener):

    def on_data(self, data):
        print(data)
        return(True)

    def on_error(self, status):
        print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["#BigData"])
