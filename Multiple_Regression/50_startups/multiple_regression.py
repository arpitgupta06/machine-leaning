import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Importing dataset
dataset = pd.read_csv('50_Startups.csv')
# print(dataset.head(5))
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(y)

# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print(x)

# Splitting the dataset into the training set and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Training the Multiple Linear Regression model on the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

