import json
import os


"""
La siguiente funcion asigna en los diccionarios correspondientes del archivo
json el valor de etiqueta de cada tweet que pertenezca a datos
@param: datos, una lista de diccionarios. A cada diccionario de que pertenezca
a la lista, se le asigna una nueva llave (si es que no la tenia antes) llamada
golden label, y un valor que depende de etiquetas.
@param etiquetas: una lista de enteros que representa si esta a favor,
en contra o neutral. Por convencion:
-2 sin etiquetar
-1 en contra
0 neutral
1 a favor

de que se haga la consulta del NAICM
"""

"""
asignar_etiqueta() es una funcion que a partir de un diccionario datos,
le asigna una etiqueta a cada llave. Para que se vea de la siguiente forma:

{"Tweet a favor de la consulta": 1, "Tweet neutro": 0}, etc. Las llaves son
los tweets y los valores son las etiquetas.

"""
def asignar_etiqueta(datos, etiquetas):
    for x in datos:
        x['golden_label'] = etiquetas[i]

# Los tweets 10 y 11 fueron clasificados para pruebas. Los tweets 2-6, 9 los clasifique yo
# y los 7, 8 paty.
with open('tweets_11.json', encoding="utf-8") as tweet_data:
    json_data = json.load(tweet_data)


# para append
vector_etiquetas_2 = [-2 for i in range(100)]
f = open("pruebas.txt", "a", encoding="utf-8")

sin_repetidos = []
rt_s = 'retweeted_status'
for i in range(100):
    os.system('cls')
    tweet = json_data['results'][i]

    if rt_s in tweet.keys():
        if 'extended_tweet' in tweet[rt_s].keys():
            if tweet[rt_s]['extended_tweet']['full_text'] in sin_repetidos:
                continue
            sin_repetidos.append(tweet[rt_s]['extended_tweet']['full_text'])
            print(len(sin_repetidos))
            print(tweet[rt_s]['extended_tweet']['full_text'])
            print("Esta a favor: 1\nEsta en contra: -1\nNeutral: 0\n")
            clasif = input("Clase: ")
            while clasif not in ['0','1','-1']:
                clasif = input("Clase: ")
            f.write(str(tweet[rt_s]['extended_tweet']['full_text']))
            f.write('\n')
            f.write('Golden label: ')
            f.write(clasif)
            f.write('\n')
        else:
            if tweet[rt_s]['text'] in sin_repetidos:
                continue
            sin_repetidos.append(tweet[rt_s]['text'])
            print(len(sin_repetidos))
            print(tweet[rt_s]['text'])
            print("Esta a favor: 1\nEsta en contra: -1\nNeutral: 0\n")
            clasif = input("Clase: ")
            while clasif not in ['0','1','-1']:
                clasif = input("Clase: ")
            f.write(str(tweet[rt_s]['text']))
            f.write('\n')
            f.write('Golden label: ')
            f.write(clasif)
            f.write('\n')
    elif 'extended_tweet' in tweet.keys():
        if tweet['extended_tweet']['full_text'] in sin_repetidos:
            continue
        sin_repetidos.append(tweet['extended_tweet']['full_text'])
        print(len(sin_repetidos))
        print(tweet['extended_tweet']['full_text'])
        print("Esta a favor: 1\nEsta en contra: -1\nNeutral: 0\n")
        clasif = input("Clase: ")
        while clasif not in ['0','1','-1']:
            clasif = input("Clase: ")
        f.write(str(tweet['extended_tweet']['full_text']))
        f.write('\n')
        f.write('Golden label: ')
        f.write(clasif)
        f.write('\n')
    else:
        if tweet['text'] in sin_repetidos:
            continue
        sin_repetidos.append(tweet['text'])
        print(len(sin_repetidos))
        print(tweet['text'])
        print("Esta a favor: 1\nEsta en contra: -1\nNeutral: 0\n")
        clasif = input("Clase: ")
        while clasif not in ['0','1','-1']:
            clasif = input("Clase: ")
        f.write(str(tweet['text']))
        f.write('\n')
        f.write('Golden label: ')
        f.write(clasif)
        f.write('\n')

    #vector_etiquetas_2[len(sin_repetidos)-1] = int(clasif)


#sin_etiquetar = [-2 for i in range(100)]
#asignar_etiqueta(json_data['results'], vector_etiquetas_2)



# Es el 2 porque el del nan es el 1, pienso yo
