# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 03:14:54 2018

@author: User
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
def extract_topn_from_vector(feature_names, sorted_items, topn):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

f = open("clasificados_2.txt", "r", encoding="utf-8")
fg = open("pruebas.txt", "r", encoding="utf-8")

# Pruebas
pruebas = fg.read()
pruebas_x = re.split('\nGolden label: -*[01]\n', pruebas)
pruebas_x.pop()
#golden_y es una variable auxiliar
y_test = []
golden_y = re.findall('Golden label: -*[01]', pruebas)
for x in golden_y:
    y_test.append(re.split('Golden label: ', x)[1])

#Entrenamiento, con el documento con etiquetados
texto_completo = f.read()
train_x = re.split('\nGolden label: -*[01]\n', texto_completo)
train_x.pop()
y_train = []
golden_y = re.findall('Golden label: -*[01]', texto_completo)
for x in golden_y:
    y_train.append(re.split('Golden label: ', x)[1])
    
#Extraemos ahora los documentos sin etiquetar(tweets que no han sido clasif)
f = open("no_clasificados.txt", "r", encoding="utf-8")
texto_completo = f.read()
sin_clasificar = re.split(r'\n<\\s>\n', texto_completo)
sin_clasificar.pop()

#Unimos los tweets sin clasificar con train_x para tener un vector de todos los
#tweets (hasta ahora, son alrededor de 5000 tweets)

for x in train_x:
    sin_clasificar.append(x)
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,1))
#X_train2 = vectorizer.fit_transform(sin_clasificar)

#X_test2 = vectorizer.transform(pruebas_x)
sin_clasificar2 = sin_clasificar[:len(y_train)]
# vectorizer es el vector tfidf de todos los datos.
#parametro max_df=0.9: si una palabra aparece mas del 90% de los documentos,
#la ignoramos (reducir parametros = reducir dimensionvc)
#parametro min_df=5: si una palabra aparece en menos de 5 documentos, la 
#ignoramos tambien. Al final, con estos datos obtenemos una matriz de 
#sparse de 805 caracteristicas
X_train = vectorizer.fit_transform(sin_clasificar2)

### La lista sin_clasificar2 es la que tiene los 1000 etiquetados
X_test = vectorizer.transform(pruebas_x)

clf = LR(multi_class="ovr", solver="liblinear").fit(X_train, y_train)

P = clf.predict(X_test)

etiquetas = ["Pro-consulta", "Neutro","Anti-consulta"]
plt.pie([y_train.count('1'), y_train.count('0'), y_train.count('-1')], labels=etiquetas, autopct='%1.1f%%',
        shadow=True, startangle=9)

X_train = vectorizer.fit_transform(sin_clasificar)
vocab = vectorizer.get_feature_names()
sorted_items = sort_coo(X_train.tocoo())
keywords = extract_topn_from_vector(vocab,sorted_items, 70)
    
datos_clase_1 = []
datos_clase_neg =[]
datos_clase_0 = []
for i in range(len(sin_clasificar2)):
    if y_train[i] == '1':
        datos_clase_1.append(sin_clasificar2[i])
    elif y_train[i] == '0':
        datos_clase_0.append(sin_clasificar2[i])
    else:
        datos_clase_neg.append(sin_clasificar2[i])
        
datos1 = vectorizer.fit_transform(datos_clase_1)
datos0 = vectorizer.fit_transform(datos_clase_0)
datos_n = vectorizer.fit_transform(datos_clase_neg)

s1 = sort_coo(datos1.tocoo())
s0 = sort_coo(datos0.tocoo())
sn = sort_coo(datos_n.tocoo())    
kw1 = extract_topn_from_vector(vocab, s1, 50)
kwn = extract_topn_from_vector(vocab, sn, 50)
kw0 = extract_topn_from_vector(vocab, s0, 50)

kf = KFold(n_splits=10)

media_lr = media_nb = 0
for train_ind, test_ind in kf.split(sin_clasificar2):
    Xtrain, Xtest = [sin_clasificar2[i] for i in train_ind], [sin_clasificar2[i] for i in test_ind]
    ytrain, ytest = [y_train[i] for i in train_ind], [y_train[i] for i in test_ind]
    
    x_entren = vectorizer.fit_transform(Xtrain)
    x_pruebas = vectorizer.transform(Xtest)
    
    reg_log = LR(multi_class="ovr", solver="liblinear")
    nb = GaussianNB()
    reg_log.fit(x_entren, ytrain)
    nb.fit(x_entren.toarray(), ytrain)
    media_lr +=reg_log.score(x_pruebas, ytest)
    media_nb += nb.score(x_pruebas.toarray(), ytest)
    
media_lr/=10
media_nb/=10
