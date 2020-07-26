# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:52:13 2020

@author: ISIL
"""
import numpy as np
import os

NUM_OF_ITERATIONS = 100000
LEARNING_RATE = 0.003
K = 3 #number of classes

path = './data'

#MULTI CLASS LOGISTIC REGRESSION

# g(z) = 1/(1+e^(-z))
def sigmoid_function(z):
    exponential = np.exp(-z)
    sig = 1.0/(1.0+exponential)
    return sig

# hΘ(x) = 1/(1+e^(-Θ.xT))
# z = (-ΘT.x) -> for this case z = (-Θ.xT)
def probability(x,theta):
    # THETA = (NUM_OF_FEATURES+1) x 1
    # x = m x (NUM_OF_FEATURES+1)
    
    #z = np.matmul(x,theta) # z = m x 1
    #to handle large arrays memory array is used
    file_name = os.path.join(path, "z.dat")
    f = np.memmap(file_name, dtype=np.float32, mode='w+', shape=(len(x), len(x[0])))
    f = np.matmul(x,theta)
    
    return sigmoid_function(f) # hΘ(x)
    

# x = m x (NUM_OF_FEATURES+1)
# y = m x 1
# THETA = (NUM_OF_FEATURES+1) x 1
    
# gradient descent WITHOUT iteration
def gradient_descent(x,y,h,theta):
    m = len(y) # number of observations    
    y_t = np.transpose(y)
    h_y = (h-y_t)
    x_t = np.transpose(x)
    g = np.matmul(x_t,h_y)
    g = LEARNING_RATE*(g/m)
    theta = theta - g
    return theta

def training(x,y,theta):

    for i in range(NUM_OF_ITERATIONS):

        h = probability(x,theta) # hΘ(x)
        theta = gradient_descent(x,y,h,theta)
    
    return theta

def decision_boundary(h):
    if(h>1):
        y=1
    elif(h<=1):
        y=0
    return y

def test(x,theta):
    
    prob = probability(x,theta)
    return prob

#converting label value to binary values by using ONE VS REST approach
def get_y(y,i,m):
    classes=['n','e','c']
    temp_y = np.zeros([1,m],dtype=int)
    
    for j in range(len(y[0])):
        if((y[0][j]).decode('UTF-8')==classes[i]):
            temp_y[0][j] = 1
        else:
            temp_y[0][j] = 0
            
    return temp_y

def accuracy(y,y_pred):
    length = len(y[0])
    
    counter = 0
    for i in range (length):
        if((y[0][i]).decode('UTF-8') == y_pred[0][i]):
            counter += 1
            
    return counter/length
            
def logistic_reg(train_x,train_y,test_x,test_y):
   
    NUM_OF_FEATURES = len(train_x[0])-1
    
    THETA = np.ones([(NUM_OF_FEATURES+1), 1], dtype = float) 
    m = len(train_x)

    H = []
    
    #training the data for #of classes times
    #h(i)(x) is calculated where i = 1,2,3
    #h(1)(x) h(2)(x) h(3)(x)
    for i in range(K):
        y_ = get_y(train_y,i,m)
        
        theta = training(train_x,y_,THETA)
        h = test(test_x,theta)
        H.append(h)

    m_t = len(test_x)
    classes=['n','e','c']
    y_predicted =  np.zeros([1,m_t],dtype=str)

    for j in range (m_t): 
        h = []
        for k in range(K):
            h.append((H[k])[j])
        
        #argmax(h(i)(x))
        max_h = max(h)
        class_index = h.index(max_h)
     
        y_predicted[0][j] = classes[class_index]
    
    acc = accuracy(test_y,y_predicted)
    return acc
