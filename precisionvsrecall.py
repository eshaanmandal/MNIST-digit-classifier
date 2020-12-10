import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
#Cross validation predict would do the cross validation and predict the score on the each cv
from sklearn.metrics import precision_recall_curve
#This would take the score we get and spit out the values for precision, recall and threshold so that we could plot it

def get_score_using_cross_validation(X_train, y_train):
   pass



X_train=pd.read_csv('prec_rec_graph_csv_data/x_train_sample.csv', header=None)
X_test=pd.read_csv('prec_rec_graph_csv_data/x_test_sample.csv', header=None)
y_train=pd.read_csv('prec_rec_graph_csv_data/y_train_5_sample.csv',header=None)
y_test=pd.read_csv('prec_rec_graph_csv_data/y_test_5_sample.csv',header=None)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

get_score_using_cross_validation(X_train, y_train)

