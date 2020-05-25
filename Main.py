import numpy as np
import pandas as pd

#Project dataset import from a csv file which are seperated by comas.
Dataset1 = pd.read_csv("ProjectData.csv", sep=",", nrows=100);

#Dividing dataset into two components that is X and y. X will contain the Column between 1 and 7. y will contain the 7 column.
X = Dataset1.iloc[:, 1:7].values
y = Dataset1.iloc[:, 7].values

#Splitting dataset as "Train Data" and "Test Data" - Cross Validation    
from sklearn.model_selection import train_test_split
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Linear Regression
from sklearn.linear_model import LinearRegression

LinearRegressionRegressor = LinearRegression()
LinearRegressionRegressor.fit(X_Train, y_Train)
y_LinearRegressionPrediction = LinearRegressionRegressor.predict(X_Test)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

print(f'MSE for Linear Regression: {mean_squared_error(y_Test, y_LinearRegressionPrediction)}') #Mean Squared Error for Linear Regression
print(f'RSME for Linear Regression: {np.sqrt(metrics.mean_squared_error(y_Test, y_LinearRegressionPrediction))}') #Root Mean Squared Error for Linear Regression
print(f'R-Squared for Linear Regression: {r2_score(y_Test, y_LinearRegressionPrediction)}') #R Squared Error for Linear Regression
print('')

#lasso
from sklearn import linear_model

Lasso = linear_model.Lasso(alpha=0.1)
Lasso.fit(X_Train,y_Train)
y_LassoPrediction = Lasso.predict(X_Test)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

print(f'MSE for Lasso: {mean_squared_error(y_Test, y_LassoPrediction)}') #Mean Squared Error for Lasso
print(f'RSME for Lasso: {np.sqrt(metrics.mean_squared_error(y_Test, y_LassoPrediction))}') #Root Mean Squared Error for Lasso
print(f'R-Squared for Lasso: {r2_score(y_Test, y_LassoPrediction)}') #R Squared Error for Lasso
print('')

#Polynomial Regression (with degree 3)
from sklearn.preprocessing import PolynomialFeatures

PolynomialRegression = PolynomialFeatures(degree = 3)
X_Polynomial = PolynomialRegression.fit_transform(X_Train)
PolynomialRegressor = LinearRegression()
PolynomialRegressor.fit(X_Polynomial, y_Train)
y_PolynomialRegressionPrediction = PolynomialRegressor.predict(PolynomialRegression.fit_transform(X_Test))

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

print(f'MSE for Polynomial Regression: {mean_squared_error(y_Test, y_PolynomialRegressionPrediction)}') #Mean Squared Error for Polynomial Regression
print(f'RSME for Polynomial Regression: {np.sqrt(metrics.mean_squared_error(y_Test, y_PolynomialRegressionPrediction))}') #Root Mean Squared Error for Polynomial Regression
print(f'R-Squared for Polynomial Regression: {r2_score(y_Test, y_PolynomialRegressionPrediction)}') #R Squared for Polynomial Regression
print('')

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

DecisionTreeRegressor = DecisionTreeRegressor(random_state = 0)
DecisionTreeRegressor.fit(X_Train, y_Train)
y_DecisionTreePrediction = DecisionTreeRegressor.predict(X_Test)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

print(f'MSE for Decision Tree: {mean_squared_error(y_Test, y_DecisionTreePrediction)}') #Mean Squared Error for Decision Tree Regression
print(f'RSME for Decision Tree: {np.sqrt(metrics.mean_squared_error(y_Test, y_DecisionTreePrediction))}') #Root Mean Squared Error for Decision Tree Regression
print(f'R-Squared for Decision Tree: {r2_score(y_Test, y_DecisionTreePrediction)}') #R Squared for Decision Tree Regression
print('')

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

RandomForestRegressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
RandomForestRegressor.fit(X_Train, y_Train)
y_RandomForestPrediction = RandomForestRegressor.predict(X_Test)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

print(f'MSE for Random Forest: {mean_squared_error(y_Test, y_RandomForestPrediction)}') #Mean Square Error for Random Forest Regression
print(f'RSME for Random Forest: {np.sqrt(metrics.mean_squared_error(y_Test, y_RandomForestPrediction))}') #Root Mean Squared Error for Random Forest Regression
print(f'R-Squared for Random Forest: {r2_score(y_Test, y_RandomForestPrediction)}') #R Squared for Random Forest Regression
print('')

#Predicted missing values beginning from 101th row of dataset
Dataset2 = pd.read_csv("ProjectData.csv", sep=",", skiprows=range(1,101))
XPrediction = Dataset2.iloc[:, 1:7].values

MissingYValues_PolynomialRegression = PolynomialRegressor.predict(PolynomialRegression.fit_transform(XPrediction))
FinalDataset = pd.DataFrame(np.insert(XPrediction, 6, MissingYValues_PolynomialRegression, axis=1))
FinalDataset.to_excel("PredictedResults.xlsx", index=True)
