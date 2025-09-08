#Dependencies for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn.datasets 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_california_housing

#importing the California Housing Price Dataset

house_price_dataset = fetch_california_housing(as_frame=True)

#loading the data set to Pandas DataFrame
house_price_dataframe = pd.DataFrame(house_price_dataset.data)

#Print first five rows of the dataset
print(house_price_dataframe.head())

# Add the target (Price) column to the Dataframe
house_price_dataframe['price'] = house_price_dataset.target
print(house_price_dataframe.head())

#checking the number of rows and coloumns in the data frame
house_price_dataframe.shape
print(house_price_dataframe.shape)

#check for the missing values
print(house_price_dataframe.isnull().sum())

#statistical measures for the datasets
print(house_price_dataframe.describe())

#checking the corelation between the dataset
correlation = house_price_dataframe.corr()

#constructing the heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
plt.show()

#Splitting the data and Target
X = house_price_dataframe.drop('price',axis=1)
Y = house_price_dataframe['price']

print(X)
print(Y)

#splitting the data into Training Data and Test Data
X_train, X_test , Y_train , Y_test = train_test_split(X , Y , test_size=.2 ,random_state= 2)
print("X.shape:", X.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("First 5 rows of X_train:")
print(X_train.head())

#loading the model
model = XGBRegressor()

#trainig the model with X_train
model.fit(X_train, Y_train)
print(model)

#accuracy for prediction on training data
training_data_prediction = model.predict(X_train)
print(training_data_prediction)

#R squared error
score_1 = metrics.r2_score(Y_train,training_data_prediction)
print("R squared error",score_1)

#Mean Absolute Error 
score_2 = metrics.mean_absolute_error(Y_train,training_data_prediction)
print("Mean absolute error",score_2)

#Visualizing the actual prices and predicted prices
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Value")
plt.show()

#prediction on test data
#accuracy for prediction on training data
test_data_prediction = model.predict(X_test)
print(test_data_prediction)

#R squared error
score_1 = metrics.r2_score(Y_test,test_data_prediction)
print("R squared error",score_1)

#Mean Absolute Error 
score_2 = metrics.mean_absolute_error(Y_test,test_data_prediction)
print("Mean absolute error",score_2)




