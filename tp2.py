# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:37:37 2018

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas
import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from skimage.io import imread,imsave

    
def compute_read_indexes(data,labels,faults):
    
    sf = sc = tp = tn = 0.0

    for ix in range(len(labels)-1):
        same_fault = faults[ix] == faults[ix+1:]

        same_cluster = labels[ix] == labels[ix+1:]

        sf += np.sum(same_fault)
        sc += np.sum(same_cluster)

        tp += np.sum(np.logical_and(same_fault,same_cluster))
        tn += np.sum(np.logical_and(np.logical_not(same_fault), np.logical_not(same_cluster)))

        total = len(labels)*(len(labels)-1)/2
        precision = tp/sc
        recall = tp/sf

        rand = (tp+tn)/total
        F1 = precision*recall*2/(precision+recall)

    return precision, recall, rand, F1, adjusted_rand_score(labels, faults), silhouette_score(data, labels)


RADIUS = 6371
data = pandas.read_csv('tp2_data.csv',delimiter=',',header=0)
longitude =data.longitude.values
latitude =data.latitude.values
depth = data.depth.values
faults = data.depth.values

nFaults = 28
nFaults = np.loadtxt('faults.csv',delimiter=',',skiprows=1)
nFaults = nFaults[-1,2]

X = RADIUS*np.cos(latitude*math.pi/180) *np.cos(longitude*math.pi/180)
Y = RADIUS*np.cos(latitude*math.pi/180) *np.sin(longitude*math.pi/180)
Z = RADIUS*np.sin(latitude*math.pi/180)

cols = np.array((X,Y,Z)).T

kmeans = KMeans(n_clusters=2).fit(cols)

MeansLabels = kmeans.labels_

MeansCentroids = kmeans.cluster_centers_

gaus = GaussianMixture(n_components=2).fit(cols)