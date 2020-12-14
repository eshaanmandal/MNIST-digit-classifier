import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
#Cross validation predict would do the cross validation and predict the score on the each cv
from sklearn.metrics import precision_recall_curve
import pickle
#This would take the score we get and spit out the values for precision, recall and threshold so that we could plot it

def get_score_using_cross_validation(X_train, y_train, model):
   y_score=cross_val_predict(model, X_train, y_train, cv=5, method="decision_function")
   print(y_score)


X_train=pd.read_csv('x_train_sample.csv', header=None)
X_test=pd.read_csv('x_test_sample.csv', header=None)
y_train=pd.read_csv('y_train_5_sample.csv',header=None)
y_test=pd.read_csv('y_test_5_sample.csv',header=None)
X_train=np.array(X_train)
X_test=np.array(X_train)
y_train=np.array(y_train).reshape(48001, -1)
y_test=np.array(y_test).reshape(12001, -1)
with open("sgd_classifier.sav", "rb") as mdl:
   sgd_clf=pickle.load(mdl)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

get_score_using_cross_validation(X_train, y_train, sgd_clf)


#also save the model in pickle in the jupyter notebook file and your work is donr with the jupyter notebook file
#After that all the graph plus score would be generated using the csv files and this program only
