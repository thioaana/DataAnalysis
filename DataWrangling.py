import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# url = "data/imports-85.data"
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(url)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers

# Identify and handle missing values
df.replace('?', np.nan, inplace = True)

missingData = df.isnull()

# Count missing values in each column
for column in missingData.columns.values.tolist():
    print(column)
    print (missingData[column].value_counts())
    print("")

# Deal with missing data
for col in ["normalized-losses","stroke", "bore", "horsepower", "peak-rpm", "price"]:
    df[col] = df[col].astype(float)

for col in ["normalized-losses", "stroke", "bore", "horsepower", "peak-rpm"]:
    df[col].replace(np.nan, df[col].mean(), inplace = True)
df["num-of-doors"].replace(np.nan, "four", inplace = True)
df["normalized-losses"] = df["normalized-losses"].astype(int)
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)  # Cause we dropped lines

# Data Standardization
df['city-L/100km'] = 235/df["city-mpg"]
df.rename(columns={"city-L/100km" : "hightway-L/100km"}, inplace = True)
print(df.head(5))

# Data Normalization
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

# Binning
df["horsepower"]=df["horsepower"].astype(int, copy=True)
plt.hist(df["horsepower"], normed = True)
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
groupNames = ["Low", "Medium", "Hight"]
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=groupNames, include_lowest=True)
print(df[['horsepower','horsepower-binned']].head(20))
print(df["horsepower-binned"].value_counts())

plt.bar(groupNames, df["horsepower-binned"].value_counts())
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

# Bins visualization
plt.hist(df["horsepower"], bins = 3)
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

# Indicator variable (or dummy variable)
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("fuel-type", axis = 1, inplace=True)

dummy1 = pd.get_dummies(df["aspiration"])
df = pd.concat([df, dummy1], axis=1)
df.drop("aspiration", axis = 1, inplace=True)
print(df.head(15))
