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


def plot_classes(labels,lon,lat, alpha=0.5, edge = 'k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("Mollweide_projection_SW.jpg")        
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

GaussianMixture(2).fit(cols)