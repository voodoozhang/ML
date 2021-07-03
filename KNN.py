# -*- coding = utf-8 -*-
# @Time : 2021/7/1 0001 15:41
# @Author : 张天赐
# @File : KNN.py
# @Software : PyCharm
import numpy as np
import operator


def creatDataSet():
    groups = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    groups.shape[0]
    return groups, labels


def filr2matrix(filrname):
    fr = open(filrname)
    infoarr = fr.readline()
    infonumb = len(infoarr)
    randmat = np.zeros((infonumb, 3))
    classlabelV = []
    index = 0
    for line in infoarr:
        line = line.strip()
        listformline = line.split('\t')
        randmat[index, :] = listformline[0:3]
        classlabelV.append((int(listformline[-1])))
        index += 1
    return randmat, classlabelV


def classifier(inx, dataset: np.ndarray, label, k):
    datasetsize = dataset.shape[0]
    diffMat = np.tile(inx, (datasetsize, 1)) - dataset
    sqdiffmat = diffMat ** 2
    sqdistance = sqdiffmat.sum(axis=1)
    distance = sqdistance ** 0.5
    sortDistance = distance.argsort()
    classcount = {}
    for i in range(k):
        voteLabel = label[sortDistance[i]]
        classcount[voteLabel] = classcount.get(voteLabel, 0) + 1
    sortcount = sorted(classcount.items(), key=lambda d: d[1], reverse=True)
    return sortcount[0][0]


def main():
    groups, labels = creatDataSet()
    inx = [1, 1]
    k = 4
    classifier(inx, groups, labels, k)


if __name__ == "__main__":
    main()
