import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
df = pd.DataFrame(dataset)
numb = len(df.columns)
print numb

X = dataset.iloc[:,:-1].values
df = pd.DataFrame(X)
numb_x = len(df.columns)
print numb_x
Y = dataset.iloc[:,-1].values
df = pd.DataFrame(Y)
numb_y = len(df.columns)
print numb_y





# Taking care of Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=-1, strategy='mean', axis = 0)
imputer = imputer.fit(X[:,:5])
X[:,:5] = imputer.transform(X[:,:5]) 



# Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state= 0)

# Fitting Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

"""X_train = X_train.reshape(1,len(X_train))
print len(X_train)"""
regressor.fit(X_train, Y_train)



# Predicting Test set results
y_pred = regressor.predict(X_test)
"""kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
scoring = 'mean_absolute_error'
#inbuilt method to calculate mean absolute error imported from sklearn
results = cross_validation.cross_val_score(regressor, X, Y, cv=kfold, scoring=scoring)
print"mean absolute error is ----> ( %.2f, %.2f )"% (results.mean(), results.std())"""
#inbuilt method to calculate mean squared error imported from numpy
print("Mean squared error: %.2f"% np.mean((regressor.predict(X_test) - Y_test)**2))
print "mean absolute error: %.2f "%np.mean(y_pred-Y_test)


sets1=[]
sets2=[]

for rows in Y_test:
    sets1.append("%.2f" % round(float(rows),2))



for rows in y_pred:
    sets2.append("%.2f" % round(float(rows),2))



total = len(sets1)
sq_means=[]
sum = 0
ab_means = 0;
#manual testing to calculate mean squared error and mean absolute error
for i in range(0, len(sets1)):
    diff = float(sets1[i])-float(sets2[i])
    #print diff
    ab_means = ab_means +diff;
    diff = diff**2
    #print diff
    sum = sum+diff;
    sq_means.append(diff)

mean = sum/total
ab_means = -1*(ab_means/total)
print "the mean is %.2f and mean absolute error is %.2f"%((mean, ab_means))

print "r^2 is  --->    %.2f    "%(r2_score(Y_test, y_pred))
