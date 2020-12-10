'''
    Here is a confusion matrix generator for a binary classifier

    By Eshaan Mandal

'''
import numpy as np
class BinaryConfusionMatrix():
    true_positive=0
    true_negative=0
    false_positive=0
    false_negative=0
    def __init__(self,bin=True):
        self.bin=bin
        
    def confusion_mtx(self, y, y_p):
        for i in range(len(y)):
            if y[i] and y_p[i]:
                self.true_positive+=1
            elif not y[i] and  not y_p[i]:
                self.true_negative+=1
            elif y[i] and not y_p[i]:
                self.false_negative+=1
            else:
                self.false_positive+=1
        return np.array([self.true_positive, self.false_negative, self.false_positive, self.true_negative]).reshape(2,2)
    def calc_recall(self):
        return self.true_positive/(self.true_positive + self.false_negative)
    def calc_precision(self):
        return self.true_positive / (self.true_positive + self.false_positive)




       
        
