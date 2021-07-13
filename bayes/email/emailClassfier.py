# -*- coding = utf-8 -*-
# @Time : 2021/7/12 0012 20:28
# @Author : 张天赐
# @File : emailClassfier.py
# @Software : PyCharm
import math
import os
import random
import re

# 读取数据集文件
import numpy as np


def textParse(paper):
    import re
    listTokens = re.split(r'\W+', paper)
    return [token.lower() for token in listTokens if len(token) > 2]


def createVocabList(words):
    vocabList = set([])
    for i in words:
        vocabList = vocabList | set(i)
    print(vocabList)

    return list(vocabList)


def createTrainMat(vocabList, doc):
    vocabMatrix = [0] * len(vocabList)
    for word in doc:
        if word in vocabList:
            vocabMatrix[vocabList.index(word)] = 1
    return vocabMatrix


def spamtest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        words = textParse(open('spam/%d.txt' % i).read())
        docList.append(words)
        classList.append(1)
        fullText.extend(words)
        words = textParse(open('ham/%d.txt' % i).read())
        docList.append(words)
        classList.append(0)
        fullText.extend(words)
    return docList, classList, fullText


def train(trainMatrix, trainClass):
    matrix_positive = np.ones(len(trainMatrix[0]))
    matrix_negative = np.ones(len(trainMatrix[0]))
    denominator_positive = 2.0
    denominator_negative = 2.0
    for i in range(len(trainMatrix)):
        if trainClass[i] == 1:
            matrix_positive += trainMatrix[i]
            denominator_positive += 1
        else:
            matrix_negative += trainMatrix[i]
            denominator_negative += 1
    pAbsuive = sum(trainClass) / len(trainClass)
    matrix_positive = matrix_positive / denominator_positive
    matrix_negative = matrix_negative / denominator_negative
    matrix_positive = np.log(matrix_positive)
    matrix_negative = np.log(matrix_negative)
    return matrix_positive, matrix_negative, pAbsuive


def bayesClassfier(matrix_positive, matrix_negative, pAbsuive, doc):
    positive = sum(doc * matrix_positive) +math.log(pAbsuive)
    negative = sum(doc * matrix_negative) + math.log(1 - pAbsuive)
    print(positive,negative)
    if positive > negative:
        return 1
    else:
        return 0


def main():
    docList, classList, fullText = spamtest()
    vocabList = createVocabList(docList)
    trainSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(createTrainMat(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    matrix_positive, matrix_negative, pAbsuive = train(np.array(trainMat), np.array(trainClass))
    res = 0
    for i in testSet:
        thisdoc = (createTrainMat(vocabList, docList[i]))
        thisdoc = np.array(thisdoc)
        bayes = bayesClassfier(matrix_positive, matrix_negative, pAbsuive, thisdoc)
        if bayes == classList[i]:
            res += 1
    print(res / len(testSet))


if __name__ == "__main__":
    main()
