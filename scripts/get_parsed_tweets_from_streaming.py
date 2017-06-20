#Este Script se conecta al Streaming de Twitter y descarga los tweets PARSEADOS
#en formato Json que contengan en el texto una palabra -ejemplo "#Podemos"
#IMPRIME solo las partes del tweet que le pidamos -p.e. "text", "user",,


#Correr desde consola con Python2 y muestra resultados por la misma consola
#python2 twitter2.py
#para escribir los resultados en un archivo:
#python2 twitter2.py > twitter2.txt


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

    def on_data(self, data):
	
        all_data = json.loads(data)
        #Si quisieramos que imprima el campo de time, debemos crear la variable:
        time = all_data["created_at"]
        tweet = all_data["text"]
        username = all_data["user"]["screen_name"]

        print((tweet))
        #Y pedir que la imprima completa
        #print((time,username,tweet))
        
	
        return True
    
    def on_error(self, status):
        print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["#Podemos"])
