# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:05:02 2017

@author: prasanthi STUDENT ID 101057215
"""

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import pandas as pd
from nltk.wsd import lesk

 
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def find_contradiction(sentence1,sentence2):
    c = 0
    for s1 in sentence1:
        for s2 in sentence2: 
            if (((s1 == "not") or (s1=="no") or (s1 == "none")) or ((s2 == "not") or (s2=="no") or (s2 == "none"))):
                #print("i have conjunction",s1)
                c = c+1
                #print(c)
                return c
 
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    contradiction=find_contradiction(word_tokenize(sentence1),word_tokenize(sentence2))
    #print("contradiction is", contradiction)
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    #print(sentence1)
    #print(sentence2)
   
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    #print(synsets1)
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    #print(synsets2)
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    #print(synsets1)
    synsets2 = [ss for ss in synsets2 if ss]
    #print(synsets2)
    score, count = 0.0, 0
    s_list = []
    count_none = 0
    for synset in synsets1:
         #print(synset)
         best = [synset.wup_similarity(ss) for ss in synsets2]
         #print(best)
         b = pd.Series(best).max()
         s_list.append(b)
    #print("similarity score is", s_list)
    scorelist = []
    for s in s_list:
       #print(s)
       if s <= 1.0:
           count_none = count_none + 1
           scorelist.append(count_none)
           #print("number of non none's are:", count_none)
           #print("number of nons are:", (len(s_list)-count_none))
           
    #print("Total number of matches with less than or equal to 1 similarity:", max(scorelist))
    #print("Total number of nones:", (len(s_list)-max(scorelist)))
    #print(sum_list(s_list))
    if contradiction == 1:
        score = sum_list(s_list)/max(scorelist) - 1
        #print("score for contradction is",score)
    else:
        score = sum_list(s_list)/max(scorelist)
        #print("score for neutral/entailment",score)
    return score

def sum_list(l):
    sum = 0
    for x in l:
        if x<= 1.0:
             sum += x
    return sum


 
#for sentence in sentences:
 #   print ("Similarity(\"%s\", \"%s\") = %s" % (focus_sentence, sentence, sentence_similarity(focus_sentence, sentence)))
    #print ("Similarity(\"%s\", \"%s\") = %s" % (sentence, focus_sentence, sentence_similarity(sentence, focus_sentence)))
    #print 



#is_anagram("The kids are playing outdoors near a man with a smile", "The young boys are playing outdoors and the man is smiling nearby")

