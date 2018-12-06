# Procesamiento del lenguaje natural: análisis de sentimientos.
Un programa sencillo para clasificar Tweets en español. En este repositorio podrás encontrar distintos Scripts, códigos fuentes en Python 3.6.x, así como datos etiquetados y no etiquetados. También texto antes de la limpieza y post-limpieza.
## Datos y antecedentes
Los Tweets que obtuvimos están relacionados de alguna manera a la Consulta que fue realizada en México en el mes de octubre de 2018, con el fin de que la sociedad pudiera mostrarse a favor o en contra de el MegaProyecto de NAIM en Texcoco. Esta consulta causó gran polémica. Es por ello que decidimos hacer un clasificador, para explicar mejor la polarización del asunto de la consulta propuesta por AMLO.
## Obtención de datos
Los datos fueron facilitados gracias a Twitter API, en donde a través de una petición amable y detallada, se nos proporcionó una cuenta personal a cada uno de los integrantes del equipo para la obtención de tweets. El siguiente script fue utilizado para obtener los datos:

```
OAUTH="XXXXXXXXXXXXXXXXXXXXXXX"
curl -X POST "https://api.twitter.com/XXXXXn" -H "Authorization: Basic $OAUTH" -H "Content-Type: application/x-www-form-urlencoded;charset=UTF-8." -d "grant_type=client_credentials"

TOKEN="XXXXXXXXXXXXXXXXXXXXX"
ENDPOINT="https://api.twitter.com/1.1/tweets/search/30day/dev.json"
NEXT="XXXXXXXX="
curl -X POST $ENDPOINT -d "{\"query\":\"aeropuerto texcoco lang:es\",\"fromDate\":\"201810260000\",\"toDate\":\"201810291159\", \"next\": \"$NEXT\"}" -H "Authorization: XXXXX $TOKEN"
```
## Archivos en el repositorio
Actualmente, podrás encontrar distintos tipos de archivo en el repositorio. Estos son:
*Extensión JSON: Son los archivos que nos proporciona Twitter API. No están etiquetados y no han pasado por un proceso de limpieza.
*Extensión txt: Estos contienen Tweets después del proceso de limpieza. Todos están previamente clasificados, a excepción de uno (se llama sin_clasificar.txt).
*Extensión py: Scripts hechos en Python 3.6.X, usando principalmente la librería SKLearn, para el entrenamiento de clasificadores, así como la extracción de características.

## Proceso y problemáticas
El obtener Tweets previamente clasificados, donde las clases son distintas a positivo o negativo, dio lugar a que optáramos por tweets sin clasificar, ya que un Tweet puede ser positivo, pero negativo de acuerdo a nuestras clases. Con ello, es posible hacer una clasificación a partir de Aprendizaje No Supervisado, donde no necesitas tener datos previamente clasificados. Sin embargo, con el único propósito de experimentación, decidimos adentrarnos al Aprendizaje Semi-Supervisado, en donde sólo necesitas una porción de los datos de entrenamiento clasificados. Nosotros mismos fuimos quienes dimos la etiqueta a los datos que actualmente, encontrarás etiquetados en este repositorio.

## Hasta el momento...
Según la referencia, de 10,000 datos, cuando 9,000 son de entrenamiento y otros 1,000 de prueba, y el 10% de los datos de entrenamiento están previamente etiquetados, se obtienen resultados ya considerables. En lo que tenemos hasta el momento, son al rededor de 500 tweets etiquetados, y cerca de 4500 no etiquetados, con 200 de prueba. Aún con ello, estamos trabajando con SKLearn para extraer las características necesarias para asegurar aprendizaje. Estas características las obtenemos del Vectorizer TFIDF, un clásico. 


## Integrantes:
*[Víctor Noriega](https://github.com/victornoriega) 
*[Nan Montaño](https://github.com/nanmon)
## Bajo la tutela de:
*[Olivia Gutú](https://github.com/oliviagutu)
