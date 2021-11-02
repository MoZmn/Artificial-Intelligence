# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print('--------------------boston dataset for regression---------------------')
print()

############# has been done just for a test on MSE calculation  ###############

#dataset for regression
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import numpy as np

bcd = load_boston()
data = pd.DataFrame(bcd.data, columns = bcd.feature_names)
data['target'] = bcd.target
x = bcd.data
y  = bcd.target

data = pd.DataFrame(x, columns = bcd.feature_names)
data['target'] = y

#creating model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model = LinearRegression()

#creating pca model
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
x_new = pca.fit_transform(x)
a = pca.explained_variance_ratio_
summ = sum(a)
print('original number of features is:', len(bcd.feature_names))
print()
print('percentage of variance coverby pca:', summ)
print()

#training the model
model.fit(x_new, y)

#evaluation of model
y_pred = model.predict(x_new)
print('MSE is: ', mean_squared_error(y, y_pred))
print()

#plotting
import matplotlib.pyplot as plt
plt.scatter(x_new[:,0], y)
plt.show()
plt.plot(np.arange(-200, 306), y_pred, color = 'purple')
plt.show()

print('-------------Banking dataset for 2 classification methods-------------')
print('---For each classifier 2 diiferent approaches for feature selection---')
print()

data = pd.read_csv(r'D:\ML & AI\786\Course Project\banking_data.csv')

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

#label encoding
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
data.drop(['day', 'month'], axis = 1, inplace = True)
data['marital']= label_encoder.fit_transform(data['marital']) 
data['education']= label_encoder.fit_transform(data['education']) 
data['default']= label_encoder.fit_transform(data['default'])
data['loan']= label_encoder.fit_transform(data['loan'])  
data['contact']= label_encoder.fit_transform(data['contact']) 
data['job']= label_encoder.fit_transform(data['job']) 
data['poutcome']= label_encoder.fit_transform(data['poutcome'])
data['housing']= label_encoder.fit_transform(data['housing'])
data['y']= label_encoder.fit_transform(data['y'])

print('-------------------------Feature selection FI-------------------------')
print()

# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
# load data
x = data.iloc[:,0:14]
y = data.iloc[:, -1]
# feature extraction
model = ExtraTreesClassifier(n_estimators = 5)
x_new = model.fit(x, y)
print("x: ", x.shape)
print("y: ", y.shape)
print()
print(model.feature_importances_)
print()
print('original number of features is:', len(data.columns) - 1)
print()

print('------------testing and training dataset after applying FI------------')
print()

import timeit
start = timeit.default_timer()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)
stop = timeit.default_timer()
print('Time: ', stop - start)  
print()

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)
print()


print('-------------------SVM and LDA classifiers algorithm------------------')
print()

print('--------------------------------- SVM --------------------------------')
print()

import timeit
start = timeit.default_timer()

from sklearn.svm import SVC   
svm = SVC(kernel='rbf')
svm.fit(x_test, y_test) 
y_pred = svm.predict(x_test)

stop = timeit.default_timer()
print('Time of SVM classification of testing set: ', stop - start)  
print()


import timeit
start = timeit.default_timer()

from sklearn.svm import SVC   
svm = SVC(kernel='rbf')
svm.fit(x_train, y_train) 
y_pred = svm.predict(x_test)

stop = timeit.default_timer()
print('Time of SVM classification of training set: ', stop - start)  
print()

# #Evaluating the Algorithm
print('confusion matrix is :', confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
from sklearn import metrics
print("Accuracy of SVM classifier:",metrics.accuracy_score(y_test, y_pred))
print()

print('--------------------------------- LDA --------------------------------')
print()

import timeit
start = timeit.default_timer()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_test, y_test)
y_pred = lda.predict(x_test)

stop = timeit.default_timer()
print('Time of LDA classification of testing set: ', stop - start)  
print()

import timeit
start = timeit.default_timer()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
y_pred = lda.predict(x_test)

stop = timeit.default_timer()
print('Time of LDA classification of training set: ', stop - start)  
print()

# #Evaluating the Algorithm
print('confusion matrix is :', confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
print('Accuracy of LDA classifier: {:.2f}'.format(lda.score(x_train, y_train)))
print()

print('-------------------------Feature selection PCA------------------------')
print()

#pca model
pca = PCA(n_components = 5)
x = data.iloc[:,0:14]
y = data.iloc[:, -1]
x_new = pca.fit_transform(x)
print("x: ", x.shape)
print("y: ", y.shape)
print("x_new: ", x_new.shape)
print()
a = pca.explained_variance_ratio_
summ = sum(a)
print('original number of features is:', len(data.columns) - 1)
print()
print('percentage of variance cover by pca:', summ)
print()

print('------------testing and training dataset after applying PCA-----------')
print()

import timeit
start = timeit.default_timer()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)
stop = timeit.default_timer()
print('Time: ', stop - start)  
print()

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)
print()


print('-------------------SVM and LDA classifiers algorithm------------------')
print()

print('--------------------------------- SVM --------------------------------')
print()

import timeit
start = timeit.default_timer()
from sklearn.svm import SVC   
svm = SVC(kernel='rbf')
svm.fit(x_test, y_test) 
y_pred = svm.predict(x_test)
stop = timeit.default_timer()

print('Time of SVM classification of testing set: ', stop - start)  
print()


import timeit
start = timeit.default_timer()

from sklearn.svm import SVC   
svm = SVC(kernel='rbf')
svm.fit(x_train, y_train) 
y_pred = svm.predict(x_test)

stop = timeit.default_timer()
print('Time of SVM classification of training set: ', stop - start)  
print()

# #Evaluating the Algorithm
print('confusion matrix is :', confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
from sklearn import metrics
print("Accuracy of SVM classifier:",metrics.accuracy_score(y_test, y_pred))
print()

print('--------------------------------- LDA --------------------------------')
print()

import timeit
start = timeit.default_timer()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_test, y_test)
y_pred = lda.predict(x_test)

stop = timeit.default_timer()
print('Time of LDA classification of testing set: ', stop - start)  
print()

import timeit
start = timeit.default_timer()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
y_pred = lda.predict(x_test)

stop = timeit.default_timer()
print('Time of LDA classification of training set: ', stop - start)  
print()

# #Evaluating the Algorithm
print('confusion matrix is :', confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
print('Accuracy of LDA classifier: {:.2f}'.format(lda.score(x_train, y_train)))
print()

print('--------------------banking dataset for regression--------------------')
print()

#dataset for regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#creating pca model
pca = PCA(n_components = 4)
x_new = pca.fit_transform(x)
print()
print("x: ", x.shape)
print("y: ", y.shape)
print("x_new: ", x_new.shape)

a = pca.explained_variance_ratio_
summ = sum(a)

print()
print('percentage of variance coverby pca:', summ)
print()

#training the model
model.fit(x_new, y)

#evaluation of model
y_pred = model.predict(x_new)
print('MSE is: ', mean_squared_error(y, y_pred))
print()

plt.scatter(x_new[:,0], y)
plt.show()
plt.plot( y, color = 'purple')
plt.show()
