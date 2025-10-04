import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



data = pd.read_csv(r"C:\Users\Shagun\Desktop\ML Regression\usa_housing_kaggle.csv")
print(data.head())

X = data[['Bedrooms','Bathrooms','SquareFeet','YearBuilt','GarageSpaces']]
Y = data['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y) # type: ignore

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train =scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
model =LinearRegression()
model.fit(X_train, Y_train)

Y_pred =model.predict(X_test)
Y_pred
Y_test

pd.DataFrame({'Actual_PRice' :Y_test, 'Predicted_Price' : Y_pred})


