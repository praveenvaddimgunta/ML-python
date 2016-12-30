# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:56:33 2016

@author: elon-pc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
df = pd.DataFrame(dataset)
co = len(df.columns)
print co
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, co-1].values
print len(X), len(y), co-1

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, co-2] = labelencoder.fit_transform(X[:, co-2])
onehotencoder = OneHotEncoder(categorical_features = [co-2])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
