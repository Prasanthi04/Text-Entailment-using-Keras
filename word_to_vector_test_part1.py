import gensim.models.word2vec as word2vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import csv
from nltk import word_tokenize
import numpy as np
from nltk import punkt, bigrams, collocations


np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

#tokenization
words = []
with open("test_data_part1.txt","r", errors='ignore') as data:
        
        dict = csv.DictReader(data, delimiter='\t')
        for row in dict:
            print(row)
            words.append(word_tokenize(row["sentence_A"]))
            words.append(word_tokenize(row["sentence_B"]))
            

'''bigram = Phrases()
bigram.add_vocab([words])            #print(words)
words_pairs = []
for i in words:        
   word_b = Phrases(i)  
   word_l = list(word_b)
   words_pairs.append(word_l)'''
     
     
          
            
#WORD2VEC
#model = word2vec.Word2Vec(words_pairs, size=50, window=5, min_count=1, workers=4)
model = word2vec.Word2Vec(words,size=50,window=5,min_count=1,workers=4)

words_vector_dict = []   
import itertools
import sys
for item in itertools.chain(words):
   #print(item)
   for i in itertools.chain(item):
      #print(i)
      filename  = open("word_to_vector1test.txt",'a')
      sys.stdout = filename
      print(i, model.wv[i], end ='\n')
      
      
      
glove_vectors_file = "word_to_vector1test.txt"
   
 
 
with open(glove_vectors_file, "r", errors='ignore') as glove:
    #next(glove)
    concat = ''.join(x.rstrip('\n') for x in glove)
    concat = concat.replace(']','\n')
   
 
file = open('bagofwordsconcattest.txt', 'w')   
file.write(concat)
file.close()
