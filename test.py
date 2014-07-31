#!/usr/bin/python

import svmMLiA
from numpy import *
import pickle

SPLITS = [[0, 1],
          [0, 2],
          [0, 3],
          [0, 4],
          [0, 5],
          [0, 6],
          [0, 7],
          [0, 8],
          [0, 9],
          [1, 2],
          [1, 3],
          [1, 4],
          [1, 5],
          [1, 6],
          [1, 7],
          [1, 8],
          [1, 9],
          [2, 3],
          [2, 4],
          [2, 5],
          [2, 6],
          [2, 7],
          [2, 8],
          [2, 9],
          [1, 8],
          [1, 9],
          [2, 3],
          [2, 4],
          [2, 5],
          [2, 6],
          [2, 7],
          [2, 8],
          [2, 9],
          [3, 4],
          [3, 5],
          [3, 6],
          [3, 7],
          [3, 8],
          [3, 9],
          [4, 5],
          [4, 6],
          [4, 7],
          [4, 8],
          [4, 9],
          [5, 6],
          [5, 7],
          [5, 8],
          [5, 9],
          [6, 7],
          [6, 8],
          [6, 9],
          [7, 8],
          [7, 9],
          [8, 9]]

'''
SPLITS = [[0,1], [0,2]]
'''


def vote(i):
  '''returns a label for item i'''

  votes = []

  for split in split_classifiers.keys():
    b, alphas, sVs, labelSV, svInd = split_classifiers[split]
    v = svmMLiA.predict(i, dataArr, split, b, alphas, sVs, labelSV, svInd)

    #print(("single vote", split, v))

    votes.append(v)

  #print(("final votes", votes))
  return most_common(votes)

def most_common(lst):
  return max(set(lst), key=lst.count)








### MAIN ###


#Turn this on to create all 45 1-1 classifiers the first time
'''
split_classifiers = {}

for split in SPLITS:
  print(("SPLIT", split))

  b,alphas,sVs,labelSV,svInd = svmMLiA.testDigits(split[0], split[1])

  split_tuple = (split[0], split[1])
  split_classifiers[split_tuple] = (b,alphas,sVs,labelSV,svInd)

print(split_classifiers.keys())

pickle.dump(split_classifiers, open("classifiers.pickle", 'w'))
'''


#'''
split_classifiers = pickle.load(open('classifiers.pickle'))

dataArr,labelArr = svmMLiA.load_mnist_all('testing')

dataArr = dataArr[:100]
labelArr = labelArr[:100]



total_guesses = 0.0
total_errors = 0.0

for i in range(len(labelArr)):
  #print(("testing element", i))
  v = vote(i)
  truth = labelArr[i]

  print('comparison: %d %d' % (v, truth))

  if v != truth:
    total_errors += 1
  total_guesses += 1

print("ERROR:", total_errors, total_guesses, total_errors/total_guesses)
#'''


