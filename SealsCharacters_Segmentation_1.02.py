#coding=utf-8
import os
from numpy.random import *
import numpy as np
import  cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import argparse
import gc

gc.collect()
from sklearn.metrics.pairwise import cosine_similarity
def getfileFromfilter(rootdir):
    list = os.listdir(rootdir)
    ReturnList=[]
    for i in range(0, len(list)):
        if list[i]!='.DS_Store':

            path = os.path.join(rootdir, list[i])
            ReturnList.append(path)
    return (ReturnList)

#===================================Binarization_using_kmeans clustering============================
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
    return im2
#===================================segmentation clustering============================

def findIndex(list,item):
    conunt = 0
    resultList=[]
    for x in list:
        if x==item:
            resultList.append(conunt)
        conunt=conunt+1
    return resultList
def preprocessing(path):
    img = np.array(cv2.imread(path))
    x_blacPixel=[]
    y_blacPixel=[]
    xy_blacPixel=[]
    for i in range(0, np.shape(img)[0]):
        for j in range(0, np.shape(img)[1]):
            if img[i][j][0] == 0:
                xy_blacPixel.append([i, j])
                x_blacPixel.append(i)
                y_blacPixel.append(j)

    return xy_blacPixel

def cluster(X,quantile):

    bandwidth = estimate_bandwidth(X ,quantile=quantile, )

    if bandwidth==0.0:
        bandwidth=0.01
    ResBandwidth=bandwidth

    ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    return labels,cluster_centers,labels_unique,n_clusters_,ResBandwidth
def clusterFinal(X,band):

    ms = MeanShift(bandwidth=band,bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    return labels,cluster_centers,labels_unique,n_clusters_,band
import random

def caculate_clusters_b(image):


    Labels_index=[]
    Bandwidth=[]
    clusters=[]
    flag=0
    #RandamBand=sorted(rand(100))
    #RandamBand = sorted([random.random() for i in range(200)])
    RandamBand=sorted( np.random.uniform(0.2 ,0.8,100))

    #RandamBand=np.arange(0.2,1,0.01)
    for i in range(0,len(RandamBand)):
        q = RandamBand[i]
        labels,cluster_centers,labels_unique,n_clusters_ ,band = cluster(preprocessing(image),q)
        Labels_index.append([band, labels])

        Bandwidth.append(band)
        clusters.append(n_clusters_)


    z = np.polyfit(Bandwidth, clusters, 3)
    p = np.poly1d(z)

    from scipy.misc import derivative
    Bandcandidate = []

    for x5 in Bandwidth:
        if derivative(p, x5, dx=1e-6, n=2) < 0.0:
            break
        Bandcandidate.append(x5)


    sortedBandcandidate = sorted(Bandcandidate, reverse=True)

    Labels_index.clear()
    for each in set(varinceCaculate(sortedBandcandidate,5)):
        print(each)
        bandwidthList = each.replace('[', '').replace(']', '').split(',')
        for eachband in bandwidthList:
            labels, cluster_centers, labels_unique, n_clusters_, band = clusterFinal(preprocessing(image),float(eachband))
            print(set(labels))
            if len(list(set(labels)))>1:
                Labels_index.append(list(labels))

    return Labels_index

def varinceCaculate(list,step):
    outputList=[list[x:x + step]for x in range(0, len(list), step)]
    varList=[]
    for eachGroup in outputList:
        varList.append([''.join(str(eachGroup)),np.std(eachGroup)])

    ranking=sorted(dict(varList).items(), key=lambda x: x[1], reverse=False)
    resultlist =[]
    for each in  ranking[:2]:
        resultlist.append(each[0])

    return resultlist


def intersec(listInput,query):
    flag=False

    for eachimage in listInput:
        inter=[]
        reinter=[]
        for each in eachimage:
            if each in query:
                inter.append(each)
        for each in eachimage:
            if each in inter:
                reinter.append(each)
        if len(inter)/len(query)>=0.90:
            flag=True
        if flag==False:
            if len(reinter) / len(eachimage) >= 0.95:
                listInput.remove(eachimage)
    if flag==False:

        listInput.append(query)

    return listInput,flag

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return int(x*200)
def mapping(image):
    img = np.array(cv2.imread(image))
    shape=np.shape(img)
    res = cv2.resize(img, (MaxMinNormalization(shape[1],max(shape),min(shape)),MaxMinNormalization(shape[0],max(shape),min(shape))))
    cv2.imwrite(image, res)
    img = np.array(cv2.imread(image))
    xy=preprocessing(image)
    Labels_index=caculate_clusters_b(image)
    cutlist=[]
    cutlist.append([[0,0]])

    for areas in Labels_index:
        for clusterN in list(set(areas)):
            pix = []
            for i in range(0,len(areas)):
                if areas[i]==clusterN:
                    pix.append(xy[i])
            cutlist,cutflag=intersec(cutlist,pix)


    print(np.shape(cutlist))
    flag=0


    for image in cutlist:
        x1=[]
        y1=[]
        for eachpix in image:
            x1.append(eachpix[1])
            y1.append(eachpix[0])

        flag=flag+1

        dstImg = img[ min(y1):max(y1),min(x1):max(x1)]


        orix=np.shape(img)[0]
        oriy=np.shape(img)[1]


        disx=np.shape(dstImg)[0]
        disy=np.shape(dstImg)[1]
        if disx!=0 and disy!=0 and oriy/disy<10and orix/disx<10 and np.average(dstImg)>0:

            cv2.imwrite('result/'+str(flag)+'.jpg', dstImg)




parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--ImageSelecter', dest='Image_path', type=str, default='Binarization2.jpg', help='ex Binarization2.jpg')
args = parser.parse_args()
print(args.Image_path)
mapping(args.Image_path)

