# AutoCop en español para ser ejecutado en local
Decarga tweets del Streaming de Twitter filtrados con una palabra clave y los clasifica en tiempo real según su sentimiento. Los sentimientos positivos/neutros o negativos han sido entrenados con un corpus de tweets políticos en español. Los resultados se escriben en pantalla y los sentimientos de los tweets clasificados con alta confianza (>0,80) se guardan en un archivo aparte para ser leídos desde un script para visualizar en directo los resultados. Si se abre el proyecto en PyCharm es de fácil ejecución, corriendo get_classified_tweets_streaming.py con entorno Python 2.7 y descargando previamente las librerias y seguidamente el script graph.py. 
Antes de ejecutar los scrips anteriores de deben entrenar los modelos para generar los archivos .pickle que de guardan en el directorio pickle_algos. Para ello se debe correr classifiers.py que lee los datos de entrenamiento de la carperta short_reviews. Autocop ha adaptado para su uso en español los scripts inicialmente desarrollados por Sentdex (https://github.com/Sentdex)

## Para citar esta herramienta:

Arcila, C.; Ortega, F.; Jiménez, J. & Trulleque, S. (2017). Análisis supervisado de sentimientos políticos en español: Clasificación en tiempo real de tweets basada en aprendizaje automático.  El Profesional de la Información, 26 (4)

## Diapositivas para el curso "Análisis de sentimiento predictivo"

https://www.dropbox.com/sh/e66vz5hdfhhdfbv/AAApnaZkIF6sPHaU-b5BAPf5a?dl=0
