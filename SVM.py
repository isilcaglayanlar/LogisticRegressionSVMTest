# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:52:13 2020

@author: ISIL
"""

from sklearn.svm import SVC

#sklearn library is used to test Support Vector Machine
def support_vector_machine(train_x,train_y,test_x,test_y):
    accuracy = (SVC(kernel = 'rbf').fit(train_x, train_y)).score(test_x, test_y) 
    return accuracy