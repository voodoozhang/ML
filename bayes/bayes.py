# -*- coding = utf-8 -*-
# @Time : 2021/7/9 0009 20:22
# @Author : 张天赐
# @File : bayes.py
# @Software : PyCharm
import numpy
import numpy as np
import math

from numpy import array


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid', 'food']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabLIst(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary" % word)
    return returnVec


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def trainNb0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 词段数
    numWords = len(trainMatrix[0])  # 合并的单词数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 有害语句的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denmo = 2.0
    p1Denmo = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denmo += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denmo += sum(trainMatrix[i])
    p1Vect = np.log(p1Num) - numpy.log(p1Denmo)
    p0Vect = np.log(p0Num) - np.log(p0Denmo)
    return p0Vect, p1Vect, pAbusive


def testingNB():
    list0Posts, listClasses = loadDataSet()  # 读取数据集
    myVocabList = createVocabLIst(list0Posts)  # 将数据集中所有单词去重合并
    trainMat = []
    for Doc in list0Posts:
        trainMat.append(setOfWords2Vec(myVocabList, Doc))
    p0v, p1v, pab = trainNb0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0v, p1v, pab):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0v, p1v, pab):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果


def main():
    testingNB()


if __name__ == "__main__":
    main()
