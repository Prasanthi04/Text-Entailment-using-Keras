# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:53:55 2017

@author: prasa
"""



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
vector_size = 50#INCREASED HIDDEN_SIZE FROM 64 TO 128

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
        #labels1 = []
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
            labels.append(row["entailment_judgment"])
                        
        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                      for x in evi_sentences])
                             
    return (hyp_sentences, evi_sentences), labels, scores #, np.array(scores)
 
data_feature_list, correct_labels, correct_score = split_data_into_scores()



from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

X = correct_score
Y = correct_labels


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=1, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)
X1 = np.array(X,dtype=float)

results = cross_val_score(estimator, X1, dummy_y, cv=kfold)
estimator.fit(X1,dummy_y)

print("Accuracy: %.2f%% " % (results.mean()*100))


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

predictions = estimator.predict(correct_scoret)
#predict_classification = []
prediction_encoder = encoder.inverse_transform(predictions)

output_DT = pd.DataFrame(data={"pair_ID":paid_idt,"entailment_judgment":prediction_encoder})
output_DT.to_csv("judgment.csv",index=False,quoting=3)

