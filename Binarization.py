#coding=utf-8
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import  cv2

import argparse
import gc
gc.collect()

def getfileFromfilter(rootdir):
    list = os.listdir(rootdir)
    ReturnList=[]
    for i in range(0, len(list)):
        if list[i]!='.DS_Store':

            path = os.path.join(rootdir, list[i])
            ReturnList.append(path)
    return (ReturnList)
#===================================Binarization_using_kmeans clustering============================
def findIndex(list,item):
    conunt = 0
    resultList=[]
    for x in list:
        if x==item:
            resultList.append(conunt)
        conunt=conunt+1
    return resultList

def drawImge(Sum_size,Indexlist):
    initList= [[255] * 3 for row in range(Sum_size)]
    for i in Indexlist:
        initList[i]=[0,0,0]
    return initList

def Binarization(path, K):
    img = cv2.imread(path)
    weight = np.shape(img)[0]
    heigh = np.shape(img)[1]

    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)
    x = []
    y = []
    z = []
    for item in Z:
        x.append(item[0])
        y.append(item[1])
        z.append(item[2])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    label1=[]

    for item in label:
        label1.append(item[0])

    for i in range(0,len(set(label1))):
        output=drawImge(np.shape(label)[0],findIndex(label1,list(set(label1))[i]))
        im2 = np.array(output).reshape(weight, heigh, 3)

        cv2.imwrite('Binarization' + str(i) + '.jpg', im2)


parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--ImageSelecter', dest='Image_path', type=str, default='test.jpg', help='Enter the file name you want to process')
parser.add_argument('--clusters', dest='K', type=int, default=3, help='Enter the number of layers you want to separate')
args = parser.parse_args()

Binarization(args.Image_path,args.K)
