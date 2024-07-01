import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
disease_df = pd.read_csv("framingham.csv")

# Drop the 'education' column and rename 'ale' to 'Sex_male'
disease_df.drop(['education'], inplace=True, axis=1)
disease_df.rename(columns={'male':'Sex_male'}, inplace=True)

# Remove NaN/NULL values
disease_df.dropna(axis=0, inplace=True)

# Print the first few rows of the dataset
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())

# Split the dataset into features (X) and target (y)
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'ysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

# Normalize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Fit the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = logreg.score(X_test, y_test)
print('Accuracy:', accuracy)
