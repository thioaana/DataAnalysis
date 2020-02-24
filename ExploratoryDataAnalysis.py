import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import  sys

url='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(url)
# print(df.head())

# 2. Analyzing Individual Feature Patterns using Visualization
print(df[["price", "bore", "stroke", "compression-ratio", "horsepower"]].corr())
print("\nStrong Linear Relationship\n", df[['highway-mpg', 'price']].corr())

# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
plt.show()

# Weak Linear Relationship
print("\nWeak Linear Relationship\n",df[['peak-rpm','price']].corr())
sns.regplot(x="peak-rpm", y="price", data=df)
plt.show()

print("\nAlso Weak Linear Relationship\n",df[['stroke','price']].corr())
sns.regplot(x="stroke", y="price", data=df)
plt.show()

# Categorical variables
sns.boxplot(x="body-style", y="price", data=df)
plt.show() # Overlaps. So Body style is not a good predictor of price

sns.boxplot(x="engine-location", y="price", data=df)
plt.show()

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.show()

# 3. Descriptive Statistical Analysis
print(df.describe()) # Does not include object type
print(df.describe(include=['object']))
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)


# 4. Basics of Grouping
dfGroupOne = df[["drive-wheels", "body-style", "price"]].groupby(["drive-wheels", "body-style"], as_index = False).mean()
print(dfGroupOne)
groupedPivot = dfGroupOne.pivot(index = "drive-wheels", columns = "body-style")
groupedPivot = groupedPivot.fillna(0) # fill missing values with 0
print(groupedPivot)

groupBodyStyle = df[["body-style", "price"]].groupby("body-style", as_index = False).mean()

plt.pcolor(groupedPivot, cmap = "RdBu")
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(groupedPivot, cmap='RdBu')

#label names
row_labels = groupedPivot.columns.levels[1]
col_labels = groupedPivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(groupedPivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(groupedPivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# 5. Correlation and Causation

# Wheel-base vs Price
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("Wheel-base vs Price : The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

# Horsepower vs Price
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("Horsepower vs Price : The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Length vs Price
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("Length vs Price : The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# Width vs Price
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("Width vs Price : The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )

# Curb-weight vs Price
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "Curb-weight vs Price : The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# Engine-size vs Price
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("Engine-size vs Price : The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Bore vs Price
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("Bore vs Price : The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value )

# City-mpg vs Price
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("City-mpg vs Price : The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# Highway-mpg vs Price
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "Highway-mpg vs Price : The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value )
