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


raw_data = df.to_numpy()