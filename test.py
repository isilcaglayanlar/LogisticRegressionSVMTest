# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:46:47 2020

@author: ISIL
"""
from preprocess import read_data, read_TFIDF, create_vectors, get_vectors, split_data,convert_list_to_nd_array
from TFIDF import compute_TFIDF
from logistic_regression import logistic_reg
from SVM import support_vector_machine


path = "./data" 

if __name__ == "__main__":
       
	#PROCESS THE DATA
    sentences, labels = read_data(path)
    TFIDF = compute_TFIDF(sentences,path) #calculate TFIDF values
    word_list, TFIDF = read_TFIDF(path) #read unique words list and TFIDF values
    create_vectors(word_list,TFIDF,path) #vectorize data
   
    NROWS = len(TFIDF)
    NCOLS = len(word_list)
    
    #vectorized corpus (premise+hypothesis)
    data = get_vectors(path,NROWS,NCOLS) #get vectorized data
   
    y = convert_list_to_nd_array("y",labels)
   
    train_x,train_y,test_x,test_y = split_data(data,y,path) #split %80 for training %20 for test
    
    #2D arrays converted into 1D arrays to use in SVM 
    train_y_1 = train_y.flatten() 
    test_y_1 = test_y.flatten() 
    
    accuracy_lg = logistic_reg(train_x,train_y,test_x,test_y)
    accuracy_svm = support_vector_machine(train_x,train_y_1,test_x,test_y_1)
   
   