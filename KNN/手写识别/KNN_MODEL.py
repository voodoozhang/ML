# -*- coding = utf-8 -*-
# @Time : 2021/7/4 0004 20:22
# @Author : 张天赐
# @File : KNN_MODEL.py
# @Software : PyCharm
import os
import numpy as np


# 将图片32*32矩阵转化为1*1024的矩阵
def img2vector(filename):
    ReturnVector = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            ReturnVector[0, 32 * i + j] = int(lineStr[j])
    return ReturnVector


def handwritingClassTest():
    # 进行数据集标注
    hwLabels = []
    trainFileList = os.listdir("trainingDigits")
    dataset = np.zeros((len(trainFileList), 1024))
    for i in range(len(trainFileList)):
        hwLabels.append(trainFileList[i].split('_')[0])
    # 制作训练集
    for i in range(len(trainFileList)):
        dataset[i, :] = (img2vector('trainingDigits/' + trainFileList[i]))
    return dataset, hwLabels


# 分类训练器
def KnnClassfier(info, label, dataset, k):
    matrix = np.tile(info, (len(label), 1)) - dataset
    smatrix = matrix ** 2
    summatrix = smatrix.sum(axis=1)
    resmatrix = summatrix ** 0.5
    labelsort = resmatrix.argsort()
    resdic = {}
    for i in range(k):
        resdic[label[labelsort[i]]] = resdic.get(label[labelsort[i]], 0) + 1
    res = sorted(resdic.items(), key=lambda k: k[1], reverse=True)[0][0]
    return res


# 测试正确率
def errorate(datasetname):
    rating = 0.0
    fr = os.listdir(datasetname)
    label = []
    for i in fr:
        label.append(i.split('_')[0])
    for i in range(len(fr)):
        info = img2vector('testDigits/' + fr[i])
        dataset, trainlabel = handwritingClassTest()
        res = KnnClassfier(info, trainlabel, dataset, 3)
        if res == label[i]:
            rating += 1.0
        else:
            print('第%i个错误' % i)
    rating = rating / len(fr)
    return rating


def main():
    # dataset, label = handwritingClassTest()
    # info = dataset[1000]
    # KnnClassfier(info, label, dataset, 400)
    print('==============开始===============')
    rate = errorate('testDigits')
    print(rate)
    print('============结束==============')


if __name__ == "__main__":
    main()
