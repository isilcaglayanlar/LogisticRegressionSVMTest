# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:50:55 2020

@author: ISIL
"""
import math
import os

# Term Frequency — Inverse Data Frequency
# w = TF*IDF

# (#word appears in a sentence/total # of words in the sentence)
def compute_TF(n,total_words):
    return n/total_words
    
    
# log(total # of sentences/# of sentences containing word)   
def compute_IDF(sentences,word,length):
    
    counter = 0    
    for sentence in sentences:
        if(word in sentence):
            counter += 1
    
   
    return math.log10(length/counter)
    
#compute TFIDF VALUE for each word in each sentences
#different sentences are assumed as different documents
#this part doesn't vectorize the sentences just calculates the TF-IDF values
def compute_TFIDF(sentences,path):
    
    file = open(os.path.join(path, "TFIDF_data.txt"), "w", encoding="utf8")
    
    TFIDF_sentences = []
    length = len(sentences)
    for i in range (length):
        TFIDF_sentence = []
        sentence = sentences[i]
        sentence_length = len(sentence)
        d = dict((x,sentence.count(x)) for x in set(sentence))
        for word,count in d.items():
            TF = compute_TF(count,sentence_length)
            IDF = compute_IDF(sentences,word,length)
            TFIDF_sentence.append(TF*IDF)
            write = word + " " + str(TF*IDF) + "\n"
            file.write(write)
            
        file.write("<>\n")
        TFIDF_sentences.append(TFIDF_sentence)
            
    file.close()
    return TFIDF_sentences
            
        