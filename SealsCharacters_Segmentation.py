#coding=utf-8
import os

import numpy as np
import  cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
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

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    return labels,cluster_centers,labels_unique,n_clusters_,ResBandwidth



def caculate_clusters_b(image):
    #Initbandwidth=0
    Endbandwidth=0
    Labels_index=[]
    Bandwidth=[]
    clusters=[]
    flag=0
    for i in range(0,50):
        q = i / 100
        labels,cluster_centers,labels_unique,n_clusters_ ,band = cluster(preprocessing(image),q)
        Labels_index.append([band, labels])

        Bandwidth.append(band)
        clusters.append(n_clusters_)

        if str(n_clusters_)==str(1)and flag==0:

            Endbandwidth=band
            flag=1
    z = np.polyfit(Bandwidth,clusters, 3)
    p = np.poly1d(z)

    from scipy.misc import derivative
    Bandcandidate=[]

    for x5 in Bandwidth:
        if derivative(p, x5, dx=1e-6,n=2)<0.0:
                break
        Bandcandidate.append(derivative(p, x5, dx=1e-6,n=2))
    result1=[]
    it = iter(Bandcandidate)
    bandi = float(it.__next__())
    for bandiplus1 in it:

        result1.append((bandiplus1 - bandi) / bandi)

    Initbandwidth=Bandwidth[result1.index(min(result1))]
    startpoint=findIndex(np.hsplit(np.array(Labels_index), 2)[0],Initbandwidth)[0]
    endpoint=findIndex(np.hsplit(np.array(Labels_index), 2)[0], Endbandwidth)[0]

    return Labels_index[startpoint:endpoint]





def mapping(image):
    img = np.array(cv2.imread(image))
    res = cv2.resize(img, (128,128))
    cv2.imwrite(image, res)
    img = np.array(cv2.imread(image))
    xy=preprocessing(image)


    Labels_index=caculate_clusters_b(image)
  
    x_init=[]
    y_init=[]
    color=[]
    for item in xy:
        x_init.append(item[0])
        y_init.append(item[1])

    for i in Labels_index[1][1]:
         color.append(i/10)


    flag=0

    for i in range(0,np.shape(Labels_index)[0]):
        if 2 in set(Labels_index[i][1]):
            for count in  set(Labels_index[i][1]):
                x1=[]
                y1=[]
                for xy1 in findIndex(Labels_index[i][1],count):
                       x1.append(xy[xy1][0])
                       y1.append(xy[xy1][1])

                flag=flag+1
                dstImg = img[ min(y1):max(y1),min(x1):max(x1)]

                cv2.imwrite('result/'+str(flag)+'.jpg', dstImg)






parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--ImageSelecter', dest='Image_path', type=str, default='Binarization2.jpg', help='ex Binarization2.jpg')
args = parser.parse_args()
print(args.Image_path)
mapping(args.Image_path)


