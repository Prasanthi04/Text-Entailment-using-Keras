

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:45:14 2017

@author: prasanthi (101057215)
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import dd
from sklearn.svm import SVR


glove_vectors_file = "bagofwordsconcat.txt"

glove_wordmap = {}
with open(glove_vectors_file, "r", errors='ignore') as glove:
    for line in glove:
        #print("hello")
        #print(line)
        name, vector = tuple(line.split(" ", 1))
        #print(name,vector)
        glove_wordmap[name] = np.fromstring(vector, sep=" ")
        #print(glove_wordmap)

#Constants setup
max_hypothesis_length, max_evidence_length = 30, 30
batch_size, vector_size, hidden_size = 128, 50, 200 #INCREASED HIDDEN_SIZE FROM 64 TO 128

lstm_size = hidden_size

weight_decay = 0.001 # CHANGED FROM 0.0001

learning_rate = 1.5

input_p, output_p = 1.0, 1.0

training_iterations_count = 120000

display_step = 10


def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    res[slices] = matrix[slices]
    return res

def sentence2sequence(sentence):
    tokens = sentence.lower().split(" ")
    rows = []
    words = []
    #Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            #print("hello")
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i-1
    return rows, words

#def split_data_into_scores():
    
    
def split_data_into_scores():

    with open("training_data_part1.txt","r") as data:
        train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        scores = []
        for row in train:
            #print(row["sentence1"])
            focus_sentence = (row["sentence_A"].lower())
            sentences = (row["sentence_B"].lower())
            sc=dd.sentence_similarity(focus_sentence,sentences)
            #print(sc)
            scores.append(sc)
            hyp_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_A"].lower())[0]))
            evi_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_B"].lower())[0]))
            labels.append(row["relatedness_score"])
            #scores.append(score_setup(row,labels))
            #print(labels)
        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                      for x in evi_sentences])
                             
    return (hyp_sentences, evi_sentences), labels, scores #, np.array(scores)
 
data_feature_list, correct_labels, correct_score = split_data_into_scores()





from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

#x = np.arange(-2,3.0,0.01)
#y = x**2 - 2*x + 1

correct_score1 = np.array(correct_score,dtype=float)
correct_labels1 = np.array(correct_labels,dtype=float)

model = Sequential()
model.add(Dense(50, activation='sigmoid', 
                input_dim=1, init='uniform'))
model.add(Dense(1, activation='linear'))
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=False)
model.compile(loss='mean_squared_error', 
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(correct_score1,correct_labels1,nb_epoch=300, batch_size = 5,verbose = 0)

#print(correct_score)

'''def predict_relatedness(score, labels):
   
    svr_len = SVR(kernel= 'linear', C=1e3)
    svr_len.fit(score,labels)
    return svr_len.predict'''


#######################

glove_vectors_file1 = "bagofwordsconcattest.txt"

glove_wordmap1 = {}
with open(glove_vectors_file1, "r", errors='ignore') as glove:
    for line in glove:
        #print("hello")
        #print(line)
        name, vector = tuple(line.split(" ", 1))
        #print(name,vector)
        glove_wordmap1[name] = np.fromstring(vector, sep=" ")
        #print(glove_wordmap)



def sentence2sequence1(sentence):
    tokens = sentence.lower().split(" ")
    rows = []
    words = []
    #Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            #print("hello")
            if word in glove_wordmap1:
                rows.append(glove_wordmap1[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i-1
    return rows, words

#def split_data_into_scores():
    
    
def split_data_into_scores1():

    with open("test_data_part1.txt","r") as data:
        train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        scores = []
        pair_id = []
        for row in train:
            #print(row["sentence1"])
            focus_sentence = (row["sentence_A"].lower())
            sentences = (row["sentence_B"].lower())
            sc=dd.sentence_similarity(focus_sentence,sentences)
            #print(sc)
            scores.append(sc)
            #print(scores)
            hyp_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_A"].lower())[0]))
            evi_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_B"].lower())[0]))
            pair_id.append(row["pair_ID"])
            #labels.append(row["relatedness_score"])
            #scores.append(score_setup(row,labels))
            #print(labels)
        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                      for x in evi_sentences])
                             
    return (hyp_sentences, evi_sentences), scores, pair_id #, np.array(scores)

data_feature_listt, correct_scoret,paid_idt = split_data_into_scores1()

correct_scoret=np.array(correct_scoret)
predict_relatedness = []
for i in range(0,4927):
    predict_relatedness.append(model.predict(np.asarray(correct_scoret[i]).reshape(1,1)))

output_DT = pd.DataFrame(data={"pair_ID":paid_idt,"Relatedness_score":predict_relatedness})
output_DT.to_csv("relatedness.csv",index=False,quoting=3)

