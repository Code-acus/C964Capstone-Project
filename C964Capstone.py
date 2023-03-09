# impoort pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler
#
# # Load data into a Pandas dataframe
# data = pd.read_csv('C:/diabetes_data.csv')
#
# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(data.drop('Outcome', axis=1),
#                                                     data['Outcome'], test_size=0.2, random_state=0)
#
# # Scale the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # Create a logistic regression model
# logreg = LogisticRegression(max_iter=1000)
#
# # Fit the model to the training data
# logreg.fit(X_train, y_train)
#
# # Make predictions on the test data
# y_pred = logreg.predict(X_test)
#
# # Evaluate the model's performance
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print('Precision:', precision_score(y_test, y_pred))
# print('Recall:', recall_score(y_test, y_pred))
# print('F1 score:', f1_score(y_test, y_pred))

# Environment Preparation:
# Libraries are initially imported to support the application. Additionally,
# a list of the associated applications and libraries are output for reference

import pandas as pd
import numpy as np
import ipywidgets as widgets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly as py
import sklearn as sk
import voila as vo
# Enables viewing of plotly graphs for offline development
import plotly.offline as pyo
pyo.init_notebook_mode()

# Load data into a Pandas dataframe
data = pd.read_csv('C:/diabetes_data.csv')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Outcome', axis=1),
                                                    data['Outcome'], test_size=0.2, random_state=0)

# Create a logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logreg.predict(X_test)

# Evaluate the model's performance
print('Operating System & Programing Language used:')
print()
print('Windows 11:           '+ 'Version 10.0, Build 22621')
print('Python:               '+ 'Version Python 3.9.13 [MSC v.1916 64 bit (AMD64)]')
print()
print()
print('Applications used in project:')
print()
print('Google Chrome:        '+ 'Version 110.0.5481.77 (Official Build) (64-bit)')
print('Anaconda Navigator:   '+ 'Version 2.3.1')
print('Jupyter Notebook:     '+ 'Version 6.4.12')
print()
print()
print('Libraries imported and associated version numbers used to support this project:')
print()
print ('Pandas:               ' + 'Version ' + pd.__version__)
print ('Numpy:                ' + 'Version ' + np.__version__)
print ('Plotly:               ' + 'Version ' + py.__version__) #Plotly:               Version 5.9.0
print ('Scikit Learn:         ' + 'Version ' + sk.__version__)
print ('Ipywidgets:           ' + 'Version ' + widgets.__version__)
print ('Voila:                ' + 'Version ' + vo.__version__)

# Data
# In this section, data is collected and later prepared to support visualizations and the machine learning algorithm.
#
# Collection
# In this stage the application's dataset retrieved from Kaggle is converted from a CSV to a Pandas DataFrame.
# Information about the DataFrame is then output for inspection.

df = pd.read_csv('C:/diabetes_data.csv')
df.info()

# This is something like what we should see:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 568 entries, 0 to 567
# Data columns (total 33 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   index     568 non-null    int64
#  1   842302    568 non-null    int64
#  2   M
