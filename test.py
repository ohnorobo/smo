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
b,alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)

ws=svmMLiA.calcWs(alphas,dataArr,labelArr)
print ws

datMat=mat(dataArr)

print datMat[0]*mat(ws)+b
print labelArr[0]

print datMat[1]*mat(ws)+b
print labelArr[1]

print datMat[2]*mat(ws)+b
print labelArr[2]
'''


#Turn this on to create all 45 1-1 classifiers the first time
'''
for split in SPLITS:
  print(("SPLIT", split))

  split_classifiers = {}

  #svmMLiA.testDigits(split[0], split[1], ('rbf', 20))
  b,alphas = svmMLiA.testDigits(split[0], split[1])

  split_tuple = (split[0], split[1])
  split_classifiers[split_tuple] = (b,alphas)

pickle.dump(split_classifiers, open("classifiers.pickle", 'w'))
'''



