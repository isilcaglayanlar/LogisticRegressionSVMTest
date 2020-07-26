# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:33:44 2020

@author: ISIL
"""

import os
import numpy as np
import random

#extracting premise and hypothesis sentences and labels from the .txt data
def read_data(path):
    
    sentences = []
    labels = []
    
    with open(os.path.join(path, "multinli_1.0_train.txt"), "r", encoding="utf8") as input_file:
        for line in input_file:
			#check for empty line
            if line.strip(): 
                index1 = 0
                index2 = 0
                
                list_index1 = []
                check_list1 = []
                
                list_index2 = []
                check_list2 = []
                
                #indexes that CAN be end of premise sentence
                list_index1.append(line.find("."))
                list_index1.append(line.find("?"))
                list_index1.append(line.find("!"))
                
                #minimum index is chosen to evaluate the index that is end of the sentence
                #since minimum index is chosen, (index = -1) condition is checked
                length1 = len(list_index1)
                end_of_sentence_count = list_index1.count(-1)
                
                #check if all indexes NOT equal to (-1)
                if(end_of_sentence_count != 3):
                    if (end_of_sentence_count > 0):
                        for i in range(length1):
                            elem = list_index1[i]
                            if(elem!=-1):
                                check_list1.append(elem)
                        
                        if((len(check_list1))>1):
                            index1 = min(check_list1)
                        else:
                            index1 = check_list1[0]
                        
                    else:
                        index1 = min(list_index1)
                
                    
                    rem_line = line[index1+1:]
                    
                    #indexes that CAN be end of hypothesis sentence
                    list_index2.append(rem_line.find("."))
                    list_index2.append(rem_line.find("?"))
                    list_index2.append(rem_line.find("!"))
                    
                    length2 = len(list_index2)
                    end_of_sentence_count = list_index2.count(-1)
                    
                    if(end_of_sentence_count != 3):
                        if (end_of_sentence_count > 0):
                            for i in range(length2):
                                elem = list_index2[i]
                                if(elem!=-1):
                                    check_list2.append(elem)
                            
                            if((len(check_list2))>1):
                                index2 = min(check_list2)
                            else:
                                index2 = check_list2[0]
                            
                        else:
                            index2 = min(list_index2)
                    
                        index2 = index1+index2
                       
                        #converting data to appropriate form
                        if(index1+1<index2):
                            x1 = line[:index1]
                            x2 = line[index1+1:index2]
                            
                            x1 = ((((x1.replace('(',"")).replace(')',"").replace(',',"")).replace('%',"")).replace('--',"")).split(" ")
                            x2 = ((((x2.replace('(',"")).replace(')',"").replace(',',"")).replace('%',"")).replace('--',"")).split(" ")
                            x1 = list(filter(None, x1))
                            x2 = list(filter(None, x2))
                            
                            
                            if(len(x2)!=0 and x2[0]=="\t"):
   
                                sentence1 = x1[1:]
                                sentence2 = x2[1:]
                                sentence1 = [each_string.lower() for each_string in sentence1]
                                sentence2 = [each_string.lower() for each_string in sentence2]

                                label = x1[0]
                                sentences.append(sentence1+sentence2)
                                labels.append(label[:-1])
    
    return sentences,labels

#read the unique words' names and all words' TFIDF values
def read_TFIDF(path):
    
    word_list = []
    sentences = []
    sentence = []
    
   
    with open(os.path.join(path, "TFIDF_data.txt"), "r", encoding="utf8") as input_file:
        
        for line in input_file:
           
			#check for empty line
            if line.strip():
                if(line=="<>\n"): #sentence seperator
                    sentences.append(sentence)
                    sentence = []
                   
                else:
                    l = (line.replace("\n","")).split(" ")
                    sentence.append(l)
                    word_list.append(l[0])
                
               
                    
    word_set = set(word_list)
    word_list.clear()
    
    for elem in word_set:
        word_list.append(elem)
    
    return word_list, sentences

#word vectors are created by using unique word list and TFIDF values of words
def create_vectors(word_list,TFIDF,path):
    

    nrows = len(TFIDF)
    ncols = len(word_list)

    file_name = os.path.join(path, "wordVectors.dat")
    f = np.memmap(file_name, dtype=np.float32, mode='w+', shape=(nrows, ncols))
    #to handle large arrays memory array is used
    
    row=0
    for sentence in TFIDF:
         col = 0
         for elem in word_list:
             check = False
             for word in sentence:
                 if (word[0]==elem):
                     f[row,col] = float(word[1])
                     check=True
                     col += 1
                     
             if(check==False):
                 f[row,col] = 0.0
                 col += 1
                 
         row +=1
    

def get_vectors(path,nrows,ncols):

    file_name = os.path.join(path, "wordVectors.dat")
    vectors = np.memmap(file_name, dtype=np.float32, mode='r',shape=(nrows, ncols))
   
    return vectors


def split_data(x,y,path):
    
    width_x = len(x)
    height_x = len(x[0])
    
    max_len = int((width_x*80)/100)
    min_len = width_x - max_len
    random_indexes = random.sample(range(0, max_len),max_len)
    
    
    file_name1 = os.path.join(path, "train_x.dat")
    train_x = np.memmap(file_name1, dtype=np.float32, mode='w+', shape=(max_len,height_x))    
    
    file_name2 = os.path.join(path, "test_x.dat")
    test_x = np.memmap(file_name2, dtype=np.float32, mode='w+', shape=(min_len,height_x))
    
    file_name3 = os.path.join(path, "train_y.dat")
    train_y = np.memmap(file_name3, dtype="S10", mode='w+', shape=(1,max_len))    
    
    file_name4 = os.path.join(path, "test_y.dat")
    test_y = np.memmap(file_name4, dtype="S10", mode='write', shape=(1,min_len))
    
    tr_counter =0
    test_counter = 0
    for i in range (width_x):
       
        if(i in random_indexes):
            train_x[tr_counter] = x[i]
            train_y[0,tr_counter] = y[0][i]
            tr_counter +=1
        else:
            test_x[test_counter] = x[i]
            test_y[0][test_counter] = y[0][i]
            test_counter +=1
    
    return train_x,train_y,test_x,test_y
    
    
#convert 1D arrays to 2D arrays
def convert_list_to_nd_array(data_type,vectors):
    
    if(data_type=="x"):
        
        width = len(vectors)
        
        height = len(vectors[0])
        array = np.ones([width,height+1], dtype=float)
        
        
        for i in range (0,width):
            for j in range (1,height+1):
                array[i][j] = vectors[i][j-1]

        return array
    
    elif(data_type=="y"):
        
        width = len(vectors)
        array = np.zeros([1,width], dtype=str)

        for i in range (width):
            
            class_val = vectors[i]
            array[0][i] = class_val[0]
            
        return array










        
