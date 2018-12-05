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
from sklearn.metrics import adjusted_rand_score,silhouette_score
from sklearn.neighbors import NearestNeighbors

def make_plot(labels,lon,lat, alpha=0.5, edge = 'k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = scipy.misc.imread("Mollweide_projection_SW.jpg")        
    plt.figure(figsize=(10,5),frameon=False)    
    x = lon/180*np.pi
    y = lat/180*np.pi
    ax = plt.subplot(111, projection="mollweide")
    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x,y)).T)
    print(np.min(np.vstack((x,y)).T,axis=0))
    print(np.min(t,axis=0))
    clims = np.array([(-np.pi,0),(np.pi,0),(0,-np.pi/2),(0,np.pi/2)])
    lims = ax.transData.transform(clims)
    plt.close()
    plt.figure(figsize=(10,5),frameon=False)    
    plt.subplot(111)
    plt.imshow(img,zorder=0,extent=[lims[0,0],lims[1,0],lims[2,1],lims[3,1]],aspect=1)        
    x = t[:,0]
    y= t[:,1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0   
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=alpha, markeredgecolor=edge)
        ix = ix+1                    
    mask = np.logical_not(nots)    
    if np.sum(mask)>0:
        plt.plot(x[mask], y[mask], '.', markersize=1, mew=1,markerfacecolor='w', markeredgecolor=edge)
    plt.axis('off')  


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

def preProcess(data):
    neighbour = NearestNeighbors(n_neighbors=4).fit(data)
    return neighbour

def arrangeCoordinates(lat,long):
    X = RADIUS*np.cos(lat*math.pi/180) *np.cos(long*math.pi/180)
    Y = RADIUS*np.cos(lat*math.pi/180) *np.sin(long*math.pi/180)
    Z = RADIUS*np.sin(lat*math.pi/180)
    return X,Y,Z
    
    
DIST = 4
RADIUS = 6371
data = pandas.read_csv('tp2_data.csv',delimiter=',',header=0)
longitude =data.longitude
latitude =data.latitude
faults = data.fault

x,y,z = arrangeCoordinates(latitude,longitude)
data=np.zeros(shape=(x.shape[0],3))
data[:,0]=x[:]
data[:,1]=y[:]
data[:,2]=z[:]

#KMEANS
bestVal=0
bestK=0
resultsk = np.zeros(shape=(98,7))
for v in range(2,100):
    kmeans = KMeans(n_clusters=v,random_state=0).fit(data)
    labels = kmeans.predict(data)
    values = compute_read_indexes(data,labels,faults)
    resultsk[v-2] = [v,values[0],values[1],values[2],values[3],values[4],values[5]]
    if((values[3]+values[4])>bestVal):
        bestVal=values[3]+values[4]
        bestK=v
kmeans=KMeans(n_clusters=bestK,random_state=0).fit(data)
labels = kmeans.predict(data)
plt.figure(figsize=(10,5),frameon=False)
plt.title("KMeans")
plt.xlabel("Nº of clusters")
plt.ylabel("validation indexes")
p1,=plt.plot(resultsk[:,0], resultsk[:,1], '-b',label="Precision");
p2,=plt.plot(resultsk[:,0], resultsk[:,2], '-r',label="Recall");
p3,=plt.plot(resultsk[:,0], resultsk[:,3], '-k',label="Rand");
p4,=plt.plot(resultsk[:,0], resultsk[:,4], '-g',label="F1");
p5,=plt.plot(resultsk[:,0], resultsk[:,5], '-y',label="ARI");
p6,=plt.plot(resultsk[:,0], resultsk[:,6], '-m',label="Silhouette");
plt.legend(handles=[p1,p2,p3,p4,p5,p6])
plt.show()
plt.close()

gaus = GaussianMixture(n_components=2).fit(cols)

neighbour = preProcess(cols)

k=4
scan = DBSCAN(eps=k,min_samples=4).fit(cols)