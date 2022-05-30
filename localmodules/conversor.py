import numpy as np

#Conversor
def dumm(X):
    L=np.zeros((19,), dtype=int).tolist()
    PV2=X[3]
    
    X = [l for l in X if l !=PV2]
    for i in range(15):#Posição 0 à 14.
        L[i]=X[i]

    x=[3,4,5] #Valores de Index.lesion.PIRADS.V2
    if PV2==x[2]:
        L[17]=1
    if PV2==x[1]:
        L[16]=1
    if PV2==x[0]:
        L[15]=1
   
    L[-1]=X[-1]
    return L

def minmax(X,data):
    L=np.zeros((19,), dtype=int).tolist()
    for i in range(5):
        min = data.min(axis=0)[i]
        max = data.max(axis=0)[i]
        if X[i] < min:
            min=X[i]
        if X[i] > max:
            max=X[i]
        std = (X[i] - min) / (max - min)
        #X_scaled = std * (max - min) + min
        L[i] = std
    for i in range(5,19):
        L[i]=X[i]
    return L 

def vectorizer(X):
    for i in range(len(X)):
        X[i]=[X[i]]
    return X