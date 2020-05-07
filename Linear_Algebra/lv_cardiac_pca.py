#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:54:23 2020

@author: ms
"""
import os
import numpy as np
from scipy import interpolate
import scipy.linalg 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def skip_comments(fp):
    while True:
        last_pos = fp.tell()
        line = fp.readline()
        if not line.strip().startswith('#') and  line.strip():
           break
    fp.seek(last_pos)
    
def getxy(line):
    line = line.split()
    return [float(line[2]), float(line[3])]

def readPoints(filepath, filename):
    with open(os.path.join(filepath,filename)) as fp:
        skip_comments(fp)
        num_points = int(fp.readline())
        skip_comments(fp)
        points = [getxy(fp.readline()) for i in range(num_points)]
    return points

def imgname_from_segfilename(filepath, filename):
    return os.path.join(filepath,filename.split(sep='.')[0]+'.bmp')


def readSegmentations(filepath):
    segmentationlist = [ readPoints(filepath,file) for file in os.listdir(filepath)
                            if file.endswith('asf')]
    return segmentationlist
    
def showImg(filename,show = False):
    plt.imshow(mpimg.imread(filename))
    plt.axis('off')
    if show:
        plt.show()
        
def getImageWH(filename):
    img = mpimg.imread(filename)
    return img.shape

def interp(points):
    points = np.vstack((points,points[0]))
    
    tck, u = interpolate.splprep([points[:,0],points[:,1]], k=3,s = 0)
    u = np.linspace(0,1,num = 50)
    interp_inner = interpolate.splev(u, tck)
    return interp_inner

def showInterp(interp_points,W=256,H=256,marker = 'r'):
    plt.plot(interp_points[0]*W,interp_points[1]*H,marker)
    plt.axis('off')
    
    
def showPoints(points,W=256,H=256, show = False,color = 'white'):
    points = np.array(points)
    plt.scatter(points[:,0]*W,points[:,1]*H,color=color,s = 1) 
    if show:
        plt.show()

def showSegImg(imgpath,points):
    W,H = getImageWH(imgpath)
    showImg(imgpath)
    showInterp(interp(points[:33]),W,H)
    showInterp(interp(points[33:]),W,H)
    showPoints(points,W,H,True)

    
def get_centroids(points):
    
    c1 = np.mean(points[:33],axis = 0)
    c2 = np.mean(points[33:],axis = 0) 
    return c1,c2

def showCentroids(centroids,W=256,H=256):
    plt.scatter(centroids[:,0,0]*W,centroids[:,0,1]*H,marker = '4',color = 'black')
    plt.scatter(centroids[:,1,0]*W,centroids[:,1,1]*H,marker = '4',color = 'black')
    plt.axis('off')

def showPCAModes(mean_centre, mode ,title = None):
    mean_center_in = mean_centre.reshape(66,-1)[:33]
    mean_center_out = mean_centre.reshape(66,-1)[33:]

    ax1 = plt.subplot(1,2,1)
    showInterp(interp(mean_center_in),marker = 'r')
    showInterp(interp(mean_center_out),marker = 'r')
    showInterp(interp(mean_center_in + mode.reshape(66,-1)[:33]),marker = 'b')
    showInterp(interp(mean_center_out + mode.reshape(66,-1)[:33]),marker = 'b')

    plt.subplot(1,2,2, sharex = ax1,sharey = ax1)
    showInterp(interp(mean_center_in),marker = 'r')
    showInterp(interp(mean_center_out),marker = 'r')
    showInterp(interp(mean_center_in - mode.reshape(66,-1)[33:]),marker = 'g')
    showInterp(interp(mean_center_out - mode.reshape(66,-1)[33:]),marker = 'g')
    if title:
        plt.suptitle(title)
    
    plt.show()

def main():
    filepath = '/home/ms/Desktop/repos/ssm_datasets/lv_cardiac/data'
    segmentationlist = readSegmentations(filepath)
    
    lv_cardiac = np.array([np.array(segment) for segment in segmentationlist])
    mean_lv_cardiac = np.mean(lv_cardiac, axis = 0)
    
    showSegImg(imgname_from_segfilename(filepath,'c4480h_s1.asf'),
               lv_cardiac[0].reshape(-1,2))
    
    mean_centroids = np.array([get_centroids(mean_lv_cardiac.reshape(-1,2))])
    centroids = np.array([get_centroids(segment) for segment in segmentationlist ])
    
    
    diff1 = centroids[:,0,:] - mean_centroids[:,0,:]
    centred1 = lv_cardiac[:,:33,:] - diff1.reshape(14,1,2)
    diff2 = centroids[:,1,:] - mean_centroids[:,1,:]
    centred2 = lv_cardiac[:,33:,:] - diff2.reshape(14,1,2)
    
    for c1,c2 in zip(centred1,centred2):
        showInterp(interp(c1),marker = 'b')
        showInterp(interp(c2))
    plt.title('Training Data LV Segmentation')
    plt.show()
    
    centred = np.concatenate((centred1.reshape(14,-1),centred2.reshape(14,-1)),axis = 1)
    _cov_mat = np.cov(centred.T)
    mean_centred = np.mean(centred, axis = 0)
    eig_val, eig_vec = scipy.linalg.eigh(_cov_mat)
    for i in range(1,5):
        mode = eig_vec[:,-i] * 3 * np.sqrt(eig_val[-i])
        showPCAModes(mean_centred,mode,"PCA Major Mode "+ str(i))
    
    
    
if __name__ == '__main__':
    main()