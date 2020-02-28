import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# path of data
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(url)
# print(df.head())
# print(df.dtypes)
# print(df[df.isna().any(axis=1)])

# 1. Linear Regression and Multiple Linear Regression
lm = LinearRegression()
X = df[["highway-mpg"]]
Y = df["price"]
lm.fit(X, Y)
Yhat = lm.predict(X)
print(lm.intercept_, lm.coef_)
