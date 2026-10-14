import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X,columns=feature_names)
df['species'] = y
print(df.head())
print("wymiary",df.shape)
print(df['species'].value_counts())


raw_data = X
print(raw_data.shape)
min = raw_data.min(axis=0)
max = raw_data.max(axis=0)
print(min)
print(max)
ranges = max-min
print(ranges)

x_norm = (raw_data - min)/ranges
print(x_norm)

rng = np.random.default_rng(42)
idx = rng.choice(150,3,replace=False)

weights = x_norm[idx].copy()
weights_start = weights.copy()
print(idx)
print(weights)
print(weights_start)


def Odleglosc(x,w):
    roznica = w-x
    kwadrat = roznica**2
    suma = np.sum(kwadrat)
    wynik = np.sqrt(suma)
    return wynik

print(Odleglosc(x_norm[0],weights[0]))
print(Odleglosc(x_norm[0],weights[1]))
print(Odleglosc(x_norm[0],weights[2]))


def Winner(x,weights):
    odleglosci = []
    for i in range(3):
        odleglosci.append(Odleglosc(x,weights[i]))
    zwyciezca = np.argmin(odleglosci)
    return zwyciezca

print(Winner(x_norm[0], weights))
print(Winner(x_norm[60], weights))
print(Winner(x_norm[120], weights))

def weights_update(x,weights,winner,alpha):
    new = weights[winner]+alpha*(x-weights[winner])
    weights[winner] = new
    return

weights = weights_start.copy()
weights_update(x_norm[0],weights,1,0.5)
print(Odleglosc(x_norm[0], weights[1]))
print(weights[0])
print(weights[2])



weights = weights_start.copy()
epochs = 50
alpha = 0.3

def train_som(data,weights,epochs,alpha):
    w = weights.copy()
    for i in range(epochs):
        for sample in(data):
            zwyciezca = Winner(sample,w)
            weights_update(sample,w,zwyciezca,alpha)
    return w
