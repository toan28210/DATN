import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# Importing the dataset
df = pd.read_csv("Advertising.csv")
df.head()
print(df)
df.info()
print(df.info())
df.columns
print(df.columns)

df.drop('Unnamed: 0', axis = 1, inplace = True)
df.head()
print(df)
sns.pairplot(df, x_vars = ['TV', 'radio', 'newspaper'], y_vars = 'sales', height = 5)
plt.show()

# Multiple Linear regression - Estimating coefficients
from sklearn.linear_model import LinearRegression
X = df.iloc[: , :-1]
y = df.iloc[: , -1]

lm1 = LinearRegression()
lm1.fit(X, y)

lm1.intercept_

lm1.coef_

list(zip(['TV', 'radio', 'newspaper'], lm1.coef_))

sns.heatmap(df.corr(), annot = True)
plt.show()

from sklearn.metrics import r2_score

lm2 = LinearRegression().fit(X[['TV', 'radio']], y)
lm2_pred = lm2.predict(X[['TV', 'radio']])

print(f"R^2 Score of our model is {r2_score(y, lm2_pred)}")

lm3 = LinearRegression().fit(X[['TV', 'radio', 'newspaper']], y)
lm3_pred = lm3.predict(X[['TV', 'radio', 'newspaper']])

print(f"R^2 Score of our model is {r2_score(y, lm3_pred)}")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df[['TV', 'radio', 'newspaper']]
y = df.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

lm4 = LinearRegression().fit(X_train, y_train)
lm4_pred = lm4.predict(X_test)

print(f"RMSE of our model is : {np.sqrt(mean_squared_error(y_test, lm4_pred))}")
print(f"R^2 of our model is : {r2_score(y_test, lm4_pred)}")
X = df[['TV', 'radio']]
y = df.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

lm5 = LinearRegression().fit(X_train, y_train)
lm5_pred = lm5.predict(X_test)

print(f"RMSE of our model is : {np.sqrt(mean_squared_error(y_test, lm5_pred))}")
print(f"R^2 of our model is : {r2_score(y_test, lm5_pred)}")

from yellowbrick.regressor import PredictionError, ResidualsPlot

visualizer = PredictionError(lm5).fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof();
df['interaction'] = df['TV'] * df['radio']

X = df[['TV', 'radio', 'interaction']]
y = df.sales

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

lm6 = LinearRegression().fit(X_train, y_train)
lm6_pred = lm6.predict(X_test)

print(f"RMSE of our model is : {np.sqrt(mean_squared_error(y_test, lm6_pred))}")
print(f"R^2 of our model is : {r2_score(y_test, lm6_pred)}")
visualizer = PredictionError(lm6).fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof();