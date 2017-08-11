# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
from datetime import datetime
import time
import math
import random
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
import csv

import struct
from scipy import spatial

from joblib import Parallel, delayed
import multiprocessing

import tfrecord_io
import kitti_shared as kitti

# xyzi[0]/rXYZ out of [-1,1]  this is reveresd
MIN_X_R = -1
MAX_X_R = 1
# xyzi[1]/rXYZ out of [-0.12,0.4097]  this is reveresd
MIN_Y_R = -0.12
MAX_Y_R = 0.4097
# Z in range [0.01, 100]
MIN_Z = 0.01
MAX_Z = 100

IMG_ROWS = 64  # makes image of 2x64 = 128
IMG_COLS = 512
PCL_COLS = 62074 # All PCL files should have rows
PCL_ROWS = 3

def image_process_subMean_divStd(img):
    out = img - np.mean(img)
    out = out / img.std()
    return out

def image_process_subMean_divStd_n1p1(img):
    out = img - np.mean(img)
    out = out / img.std()
    out = (2*((out-out.min())/(out.max()-out.min())))-1
    return out

def odometery_writer(ID,
                     pclList,
                     imgDepthList,
                     tMatTargetList,
                     tfRecFolder,
                     numTuples):
    '''
    ID: python list with size 2
    pclA, pclB: numpy matrix of size nx4, mx4
    imgDepthA, imgDepthB: numpy matrix of size 128x512
    tmatTarget: numpy matrix of size 4x4
    tfRecFolder: folder name
    '''
    pclNumpy = np.asarray(pclList)
    pclNumpy = np.swapaxes(np.swapaxes(pclNumpy,0,1),1,2) # convert d x h x w -> h x w x d
    imgDepthNumpy = np.asarray(imgDepthList)
    imgDepthNumpy = np.swapaxes(np.swapaxes(imgDepthNumpy,0,1),1,2) # convert d x h x w -> h x w x d
    tMatTargetNumpy = np.asarray(tMatTargetList)
    tMatTargetNumpy = np.swapaxes(tMatTargetNumpy,0,1) # convert d x n -> n x d
    filename = str(ID[0]) + "_" + str(ID[1]) + "_" + str(ID[2])
    tfrecord_io.tfrecord_writer_ntuple(ID,
                                pclNumpy,
                                imgDepthNumpy,
                                tMatTargetNumpy,
                                tfRecFolder,
                                numTuples,
                                filename)
    return
##################################
def _zero_pad(xyzi, num):
    '''
    Append xyzi with num 0s to have unified pcl length of 
    '''
    if num < 0:
        print("xyzi shape is", xyzi.shape)
        print("MAX PCL_COLS is", PCL_COLS)
        raise ValueError('Error... PCL_COLS should be the unified max of the whole system')
    elif num > 0:
        pad = np.zeros([xyzi.shape[0], num], dtype=float)
        xyzi = np.append(xyzi, pad, axis=1)
    # if num is 0 -> do nothing
    return xyzi

def _add_corner_points(xyzi, rXYZ):
    '''
    MOST RECENT CODE A10333
    Add MAX RANGE for xyzi[0]/rXYZ out of [-1,1]
    Add MIN RANGE for xyzi[1]/rXYZ out of [-0.12,0.4097]
    '''
    ### Add Two corner points with z=0 and x=rand and y calculated based on a, For max min locations
    ### Add Two min-max depth point to correctly normalize distance values
    ### Will be removed after histograms

    xyzi = np.append(xyzi, [[MIN_Y_R], [MIN_Y_R], [0]], axis=1) # z not needed
    rXYZ = np.append(rXYZ, 1)
    xyzi = np.append(xyzi, [[MAX_X_R], [MAX_Y_R], [0]], axis=1) # z not needed
    rXYZ = np.append(rXYZ, 1)
    #z = 0.0
    #x = 2.0
    #a = 0.43
    #y = np.sqrt(((a*a)*((x*x)*(x*x)))/(1-(a*a)))
    #xyzi = np.append(xyzi, [[x], [y], [z]], axis=1)
    #rXYZ = np.append(rXYZ, np.sqrt((x*x)+(y*y)+(z*z)))
    #x = -2.0
    #a = -0.1645
    #y = np.sqrt(((a*a)*((x*x)*(x*x)))/(1-(a*a)))
    #xyzi = np.append(xyzi, [[x], [y], [z]], axis=1)
    #rXYZ = np.append(rXYZ, np.sqrt((x*x)+(y*y)+(z*z)))

    xyzi = np.append(xyzi, [[0], [0], [MIN_Z]], axis=1)
    rXYZ = np.append(rXYZ, MIN_Z*MIN_Z)
    xyzi = np.append(xyzi, [[0], [0], [MAX_Z]], axis=1)
    rXYZ = np.append(rXYZ, MAX_Z*MAX_Z)
    return xyzi, rXYZ

def _remove_corner_points(xyzi):
    xyzi = np.delete(xyzi, xyzi.shape[1]-1,1)
    xyzi = np.delete(xyzi, xyzi.shape[1]-1,1)
    xyzi = np.delete(xyzi, xyzi.shape[1]-1,1)
    xyzi = np.delete(xyzi, xyzi.shape[1]-1,1)
    return xyzi

def _get_plane_view(xyzi, rXYZ):
    ### Flatten to a plane
    # 0 left-right, 1 is up-down, 2 is forward-back
    xT = (xyzi[0]/rXYZ).reshape([1, xyzi.shape[1]])
    yT = (xyzi[1]/rXYZ).reshape([1, xyzi.shape[1]])
    zT = rXYZ.reshape([1, xyzi.shape[1]])
    planeView = np.append(np.append(xT, yT, axis=0), zT, axis=0)
    return planeView
def _normalize_Z_weighted(z):
    '''
    As we have higher accuracy measuring closer points
    map closer points with higher resolution
    0---20---40---60---80---100
     40%  25%  20%  --15%--- 
    '''
    for i in range(0, z.shape[0]):
        if z[i] < 20:
            z[i] = (0.4*z[i])/20
        elif z[i] < 40:
            z[i] = ((0.25*(z[i]-20))/20)+0.4
        elif z[i] < 60:
            z[i] = (0.2*(z[i]-40))+0.65
        else:
            z[i] = (0.15*(z[i]-60))+0.85
    return z
def _make_image(depthview, rXYZ):
    '''
    Get depthview and generate a depthImage
    '''
    '''
    We found that the plane slop is in between [ -0.1645 , 0.43 ] # [top, down]
    So any point beyond this should be trimmed.
    And all points while converting to depthmap should be grouped in this range for Y
    Regarding X, we set all points with z > 0. This means slops for X are inf
    
    We add 2 points to the list holding 2 corners of the image plane
    normalize points to chunks and then remove the auxiliary points

    [-9.42337227   14.5816927   30.03821182  $ 0.42028627  $  34.69466782]
    [-1.5519526    -0.26304439  0.28228107   $ -0.16448526 $  1.59919727]
    '''
    ### Flatten to a plane
    depthview = _get_plane_view(depthview, rXYZ)
    ##### Project to image coordinates using histograms
    ### Add maximas and minimas. Remove after histograms ----
    depthview, rXYZ = _add_corner_points(depthview, rXYZ)
    # Normalize to 0~1
    depthview[0] = (depthview[0] - np.min(depthview[0]))/(np.max(depthview[0]) - np.min(depthview[0]))
    depthview[1] = (depthview[1] - np.min(depthview[1]))/(np.max(depthview[1]) - np.min(depthview[1]))
    # there roughly should be 64 height bins group them in 64 clusters
    xHist, xBinEdges = np.histogram(depthview[0], 512)
    yHist, yBinEdges = np.histogram(depthview[1], 64)
    xCent = np.ndarray(shape=xBinEdges.shape[0]-1)
    for i in range(0, xCent.shape[0]):
        xCent[i] = (xBinEdges[i]+xBinEdges[i+1])/2
    yCent = np.ndarray(shape=yBinEdges.shape[0]-1)
    for i in range(0, yCent.shape[0]):
        yCent[i] = (yBinEdges[i]+yBinEdges[i+1])/2
    # make image of size 128x512 : 64 -> 128 (double sampling the height)
    depthImage = np.zeros(shape=[128, 512])
    # normalize range values
    #depthview[2] = (depthview[2]-np.min(depthview[2]))/(np.max(depthview[2])-np.min(depthview[2]))
    depthview[2] = _normalize_Z_weighted(depthview[2])
    depthview[2] = 1-depthview[2]
    ### Remove maximas and minimas. -------------------------
    depthview = _remove_corner_points(depthview)
    # sorts ascending
    idxs = np.argsort(depthview[2], kind='mergesort')
    # assign range to pixels
    for i in range(depthview.shape[1]-1, -1, -1): # traverse descending
        yidx = np.argmin(np.abs(yCent-depthview[1, idxs[i]]))
        xidx = np.argmin(np.abs(xCent-depthview[0, idxs[i]]))
        # hieght is 2x64
        yidx = yidx*2
        depthImage[yidx, xidx] = depthview[2, idxs[i]]
        depthImage[yidx+1, xidx] = depthview[2, idxs[i]]
    return depthImage
def get_depth_image_pano_pclView(xyzi, height=1.6):
    '''
    Gets a point cloud
    Keeps points higher than 'height' value and located on the positive Z=0 plane
    Returns correstempMaxponding depthMap and pclView
    '''
    '''
    MOST RECENT CODE A10333
    remove any point who has xyzi[0]/rXYZ out of [-1,1]
    remove any point who has xyzi[1]/rXYZ out of [-0.12,0.4097]
    '''
    #print('0', max(xyzi[0]), min(xyzi[0])) # left/right (-)
    #print('1', max(xyzi[1]), min(xyzi[1])) # up/down (-)
    #print('2', max(xyzi[2]), min(xyzi[2])) # in/out
    rXYZ = np.linalg.norm(xyzi, axis=0)
    xyzi = xyzi.transpose()
    first = True
    for i in range(xyzi.shape[0]):
        # xyzi[i][2] >= 0 means all the points who have depth larger than 0 (positive depth plane)
        if (xyzi[i][2] >= 0) and (xyzi[i][1] < height) and (rXYZ[i] > 0) and (xyzi[i][0]/rXYZ[i] > -1) and (xyzi[i][0]/rXYZ[i] < 1) and (xyzi[i][1]/rXYZ[i] > -0.12) and (xyzi[i][1]/rXYZ[i] < 0.4097): # frontal view & above ground & x in range & y in range
            if first:
                pclview = xyzi[i].reshape(xyzi.shape[1], 1)
                first = False
            else:
                pclview = np.append(pclview, xyzi[i].reshape(xyzi.shape[1], 1), axis=1)
    rPclview = np.linalg.norm(pclview, axis=0)
    depthImage = _make_image(pclview, rPclview)
    pclview = _zero_pad(pclview, PCL_COLS-pclview.shape[1])
    return depthImage, pclview
################################
def transform_pcl_2_origin(xyzi_col, tMat2o):
    '''
    pointcloud i, and tMat2o i to origin
    '''
    intensity_col = xyzi_col[3]
    xyz1 = xyzi_col.copy()
    xyz1[3] *= 0
    xyz1[3] += 1
    xyz1 = np.matmul(tMat2o, xyz1)
    xyz1[3] = intensity_col
    return xyz1
################################
def _get_tMat_A_2_B(tMatA2o, tMatB2o):
    '''
    tMatA2o A -> O (source pcl is in A), tMatB2o B -> O (target pcl will be in B)
    return tMat A -> B
    '''
    # tMatA2o: A -> Orig
    # tMatB2o: B -> Orig ==> inv(tMatB2o): Orig -> B
    # inv(tMatB2o) * tMatA2o : A -> B
    tMatA2o = np.append(tMatA2o, [[0, 0, 0, 1]], axis=0)
    tMatB2o = np.append(tMatB2o, [[0, 0, 0, 1]], axis=0)
    tMatA2B = np.matmul(np.linalg.inv(tMatB2o), tMatA2o)
    tMatA2B = np.delete(tMatA2B, tMatA2B.shape[0]-1, 0)
    return tMatA2B
def _get_tMat_B_2_A(tMatA2o, tMatB2o):
    '''
    tMatA2o A -> O (pcl is in A), tMatB2o B -> O (pcl will be in B)
    return tMat B -> A
    '''
    # tMatA2o: A -> Orig ==> inv(tMatA2o): Orig -> A
    # tMatB2o: B -> Orig 
    # inv(tMatA2o) * tMatB2o : B -> A
    tMatA2o = np.append(tMatA2o, [[0, 0, 0, 1]], axis=0)
    tMatB2o = np.append(tMatB2o, [[0, 0, 0, 1]], axis=0)
    tMatB2A = np.matmul(np.linalg.inv(tMatA2o), tMatB2o)
    tMatB2A = np.delete(tMatB2A, tMatB2A.shape[0]-1, 0)
    return tMatB2A

def _get_tMat_B_2_O(tMatA2o, tMatA2B):
    '''
    tMatA2o A -> O (target pcl will be in O), tMatA2B A -> B (source pcl is in B)
    return tMat B -> O
    '''
    # tMatA2o: A -> Orig
    # tMatA2B: A -> B ==> inv(tMatA2B): B -> A
    # tMatA2o * inv(tMatA2B) : B -> O
    tMatA2o = np.append(tMatA2o, [[0, 0, 0, 1]], axis=0)
    tMatA2B = np.append(tMatA2B, [[0, 0, 0, 1]], axis=0)
    tMatB2o = np.matmul(tMatA2o, np.linalg.inv(tMatA2B))
    tMatB2o = np.delete(tMatB2o, tMatB2o.shape[0]-1, 0)
    return tMatB2o

def _get_3x4_tmat(poseRow):
    return poseRow.reshape([3,4])
def _get_pcl_XYZ(filePath):
    '''
    Get a bin file address and read it into a numpy matrix
    Converting LiDAR coordinate system to Camera coordinate system for pose transform
    '''
    f = open(filePath, 'rb')
    i = 0
    j = 0
    pclpoints = list()
    # Camera: x = right, y = down, z = forward
    # Velodyne: x = forward, y = left, z = up
    # GPS/IMU: x = forward, y = left, z = up
    # Velodyne -> Camera (transformation matrix is in camera order)
    #print('Reading X = -y,         Y = -z,      Z = x,     i = 3')
    #               0 = left/right, 1 = up/down, 2 = in/out
    while f.readable():
        xyzi = f.read(4*4)
        if len(xyzi) == 16:
            row = struct.unpack('f'*4, xyzi)
            if j%1 == 0:
                pclpoints.append([-1*row[1], -1*row[2], row[0]]) # row[3] is intensity and not used
                i += 1
        else:
            #print('num pclpoints =', i)
            break
        j += 1
        #if i == 15000:
        #    break
    f.close()
    # convert to numpy
    xyzi = np.array(pclpoints, dtype=np.float32)
    return xyzi.transpose()



def process_dataset(startTime, durationSum, pclFolderList, seqIDs, pclFilenamesList, poseFileList, tfRecFolder,  numTuples, i):
    '''
    pclFilenames: list of pcl file addresses
    poseFile: includes a list of pose files to read
    point cloud is moved to i+1'th frame:
        tMatAo (i): A->0
        tMatBo (i+1): B->0
        tMatAB (target): A->B  (i -> i+1) 
    '''
    '''
    Calculate the Yaw, Pitch, Roll from Rotation Matrix
    and extraxt dX, dY, dZ
    use them to train the network
    '''
    seqID = seqIDs[i]
    print("SeqID started : ", seqID)

    pclFolder = pclFolderList[i]
    pclFilenames = pclFilenamesList[i]
    poseFile = poseFileList[i]

    xyziList = list()
    imgDepthList = list()
    poseB2AList = list()
    poseX20List = list()
    # pop the first in Tuples and append last as numTuple
    for j in range(numTuples-1, len(pclFilenames)-1):
        if (j == numTuples-1):
            # get 0
            xyzi = _get_pcl_XYZ(pclFolder + pclFilenames[0])
            imgDepth, xyzi = get_depth_image_pano_pclView(xyzi)
            xyziList.append(xyzi)
            imgDepthList.append(imgDepth)
            poseX20List.append(_get_3x4_tmat(poseFile[0]))
            # read rest numTuples once
            for i in range(1, numTuples):
                xyzi = _get_pcl_XYZ(pclFolder + pclFilenames[i])
                imgDepth, xyzi = get_depth_image_pano_pclView(xyzi)
                xyziList.append(xyzi)
                imgDepthList.append(imgDepth)
                poseX20List.append(_get_3x4_tmat(poseFile[i]))
                # get target pose  B->A also changes to abgxyz : get abgxyzb-abgxyza
                pose_B2A = _get_tMat_B_2_A(poseX20List[i-1], poseX20List[i])
                abgxyzB2A = kitti._get_params_from_tmat(pose_B2A)
                poseB2AList.append(abgxyzB2A)
        else:
            xyziList.pop(0)
            imgDepthList.pop(0)
            poseB2AList.pop(0)
            poseX20List.pop(0)
            # get i
            xyzi = _get_pcl_XYZ(pclFolder + pclFilenames[j])
            imgDepth, xyzi = get_depth_image_pano_pclView(xyzi)
            xyziList.append(xyzi)
            imgDepthList.append(imgDepth)
            poseX20List.append(_get_3x4_tmat(poseFile[j]))
            # get target pose  B->A also changes to abgxyz : get abgxyzb-abgxyza
            pose_B2A = _get_tMat_B_2_A(poseX20List[numTuples-2], poseX20List[numTuples-1])
            abgxyzB2A = kitti._get_params_from_tmat(pose_B2A)
            poseB2AList.append(abgxyzB2A)

        #
        fileID = [int(seqID)+100, (j-(numTuples-1))+100000, j+100000]
        odometery_writer(fileID,# 3 ints
                         xyziList,# ntuplex3xPCL_COLS
                         imgDepthList,# ntuplex128x512
                         poseB2AList,# (ntuple-1)x6
                         tfRecFolder,
                         numTuples)
    print("SeqID completed : ", seqID)
    return
################################
def _get_pose_data(posePath):
    return np.loadtxt(open(posePath, "r"), delimiter=" ")
def _get_pcl_folder(pclFolder, seqID):
    return pclFolder + seqID + '/' + 'velodyne/'
def _get_pose_path(poseFolder, seqID):
    return poseFolder + seqID + ".txt"
def _get_file_names(readFolder):
    filenames = [f for f in listdir(readFolder) if (isfile(join(readFolder, f)) and "bin" in f)]
    filenames.sort()
    return filenames

def prepare_dataset(datasetType, pclFolder, poseFolder, seqIDs, tfRecFolder, numTuples=1):
    durationSum = 0
    # make a list for each sequence
    pclFolderPathList = list()
    pclFilenamesList = list()
    poseFileList = list()
    print("Arranging filenames")
    for i in range(len(seqIDs)):
        posePath = _get_pose_path(poseFolder, seqIDs[i])
        poseFile = _get_pose_data(posePath)
        #print(posePath)

        pclFolderPath = _get_pcl_folder(pclFolder, seqIDs[i])
        pclFilenames = _get_file_names(pclFolderPath)
        
        poseFileList.append(poseFile)
        pclFolderPathList.append(pclFolderPath)
        pclFilenamesList.append(pclFilenames)
    
    print("Starting datawrite")
    startTime = time.time()
    num_cores = multiprocessing.cpu_count() - 2
    #for j in range(0,len(pclFilenames)-numTuples):
    #    process_dataset(startTime, durationSum, pclFolderPath, seqIDs[i], pclFilenames, poseFile, tfRecFolder, numTuples, j)
    Parallel(n_jobs=num_cores)(delayed(process_dataset)(startTime, durationSum, pclFolderPathList, seqIDs, pclFilenamesList, poseFileList, tfRecFolder, numTuples, j) for j in range(0,len(seqIDs)))
    #for i in range(0,len(pclFilenames)-numTuples):
    #    print(shapes[i])

    print('Done')

### ################################
### ################################
### ################################
### ################################
### ################################
### def _get_xy_maxmins(depthview, rXYZ):
###     '''
###     Get depthview and generate a depthImage
###     '''
###     '''
###     We found that the plane slop is in between [ -0.1645 , 0.43 ] # [top, down]
###     So any point beyond this should be trimmed.
###     And all points while converting to depthmap should be grouped in this range for Y
###     Regarding X, we set all points with z > 0. This means slops for X are inf
###     
###     We add 2 points to the list holding 2 corners of the image plane
###     normalize points to chunks and then remove the auxiliary points
### 
###     [-9.42337227   14.5816927   30.03821182  $ 0.42028627  $  34.69466782]
###     [-1.5519526    -0.26304439  0.28228107   $ -0.16448526 $  1.59919727]
### 
###     '''
###     ### Flatten to a plane
###     depthview = _get_plane_view(depthview, rXYZ)
###     ##### Project to image coordinates using histograms
###     # 0 - max (not necesseary)
###     xmin = np.min(depthview[0])
###     xmax = np.max(depthview[0])
###     ymin = np.min(depthview[1])
###     ymax = np.max(depthview[1])
###     return xmin, xmax, ymin, ymax 
### 
### def get_max_mins_pclView(xyzi, height=1.6):
###     '''
###     Gets a point cloud
###     Keeps points higher than 'height' value and located on the positive Z=0 plane
###     Returns corresponding depthMap and pclView
###     '''
###     #print('0', max(xyzi[0]), min(xyzi[0])) # left/right (-)
###     #print('1', max(xyzi[1]), min(xyzi[1])) # up/down (-)
###     #print('2', max(xyzi[2]), min(xyzi[2])) # in/out
###     rXYZ = np.sqrt(np.multiply(xyzi[0], xyzi[0])+
###                    np.multiply(xyzi[1], xyzi[1])+
###                    np.multiply(xyzi[2], xyzi[2]))
###     xyzi = xyzi.transpose()
###     first = True
###     for i in range(xyzi.shape[0]):
###         # xyzi[i][2] >= 0 means all the points who have depth larger than 0 (positive depth plane)
###         if (xyzi[i][2] >= 0) and (xyzi[i][1] < height): # frontal view, above ground
###             if first:
###                 pclview = xyzi[i].reshape(1, 4)
###                 first = False
###             else:
###                 pclview = np.append(pclview, xyzi[i].reshape(1, 4), axis=0)
###     pclview = pclview.transpose()
###     rXYZ = np.sqrt(np.multiply(pclview[0], pclview[0])+
###                    np.multiply(pclview[1], pclview[1])+
###                    np.multiply(pclview[2], pclview[2]))
###     xmin, xmax, ymin, ymax = _get_xy_maxmins(pclview, rXYZ)
###     return xmin, xmax, ymin, ymax
### 
### def process_maxmins(startTime, durationSum, pclFolder, pclFilenames, poseFile, i):
###     # get i
###     xyzi_A = _get_pcl_XYZ(pclFolder + pclFilenames[i])
###     pose_Ao = _get_correct_tmat(poseFile[i])
###     xmin, xmax, ymin, ymax = get_max_mins_pclView(xyzi_A)
###     return xmin, xmax, ymin, ymax
### 
### def find_max_mins(datasetType, pclFolder, poseFolder, seqIDs):
###     durationSum = 0
###     for i in range(len(seqIDs)):
###         print('Procseeing ', seqIDs[i])
###         posePath = _get_pose_path(poseFolder, seqIDs[i])
###         poseFile = _get_pose_data(posePath)
###         print(posePath)
### 
###         pclFolderPath = _get_pcl_folder(pclFolder, seqIDs[i])
###         pclFilenames = _get_file_names(pclFolderPath)
###         startTime = time.time()
###         num_cores = multiprocessing.cpu_count()
###         xmaxs = -100000.0
###         xmins = 1000000.0
###         ymaxs = -100000.0
###         ymins = 1000000.0
###         for j in range(0,100):#len(pclFilenames)-1):
###             tempXmin, tempXmax, tempYmin, tempYmax = process_maxmins(startTime, durationSum, pclFolderPath, pclFilenames, poseFile, j)
###             if xmaxs < tempXmax:
###                 xmaxs = tempXmax
###             if xmins > tempXmin:
###                 xmins = tempXmin
###             if ymaxs < tempYmax:
###                 ymaxs = tempYmax
###             if ymins > tempYmin:
###                 ymins = tempYmin
###         print('X min, X max: ', xmins, xmaxs)
###         print('Y min, Y max: ', ymins, ymaxs)
###     print('Done')
### 
### ################################
### ################################
### ################################
### ################################
### ################################
### ################################
### ################################
### def get_max_pclrows(xyzi, height=1.6):
###     '''
###     Gets a point cloud
###     Keeps points higher than 'height' value and located on the positive Z=0 plane
###     Returns corresponding depthMap and pclView
###     '''
###     '''
###     MOST RECENT CODE A10333
###     remove any point who has xyzi[0]/rXYZ out of [-1,1]
###     remove any point who has xyzi[1]/rXYZ out of [-0.12,0.4097]
###     '''
###     #print('0', max(xyzi[0]), min(xyzi[0])) # left/right (-)
###     #print('1', max(xyzi[1]), min(xyzi[1])) # up/down (-)
###     #print('2', max(xyzi[2]), min(xyzi[2])) # in/out
###     rXYZ = np.sqrt(np.multiply(xyzi[0], xyzi[0])+
###                    np.multiply(xyzi[1], xyzi[1])+
###                    np.multiply(xyzi[2], xyzi[2]))
###     xyzi = xyzi.transpose()
###     first = True
###     for i in range(xyzi.shape[0]):
###         # xyzi[i][2] >= 0 means all the points who have depth larger than 0 (positive depth plane)
###         if (xyzi[i][2] >= 0) and (xyzi[i][1] < height) and (xyzi[i][0]/rXYZ[i] > -1) and (xyzi[i][0]/rXYZ[i] < 1) and (xyzi[i][1]/rXYZ[i] > -0.12) and (xyzi[i][1]/rXYZ[i] < 0.4097): # frontal view & above ground & x in range & y in range
###             if first:
###                 pclview = xyzi[i].reshape(1, 4)
###                 first = False
###             else:
###                 pclview = np.append(pclview, xyzi[i].reshape(1, 4), axis=0)
###     rows = pclview.shape[0]
###     return rows
### 
### def process_pclmaxs(startTime, durationSum, pclFolder, pclFilenames, poseFile, i):
###     # get i
###     xyzi_A = _get_pcl_XYZ(pclFolder + pclFilenames[i])
###     pose_Ao = _get_correct_tmat(poseFile[i])
###     pclmax = get_max_pclrows(xyzi_A)
###     return pclmax
### 
### def find_max_PCL(datasetType, pclFolder, poseFolder, seqIDs):
###     durationSum = 0
###     for i in range(len(seqIDs)):
###         print('Processeing ', seqIDs[i])
###         posePath = _get_pose_path(poseFolder, seqIDs[i])
###         poseFile = _get_pose_data(posePath)
###         print(posePath)
### 
###         pclFolderPath = _get_pcl_folder(pclFolder, seqIDs[i])
###         pclFilenames = _get_file_names(pclFolderPath)
###         startTime = time.time()
###         num_cores = multiprocessing.cpu_count()
###         pclmaxList = Parallel(n_jobs=num_cores)(delayed(process_pclmaxs)(startTime, durationSum, pclFolderPath, pclFilenames, poseFile, j) for j in range(0,len(pclFilenames)-1))
###         print('Max', np.max(pclmaxList))
###     print('Done')

############# PATHS
import os
def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

pclPath = '../Data/kitti/pointcloud/'
posePath = '../Data/kitti/poses/'
seqIDtrain = ['00', '01', '02', '03', '04', '05', '06', '07', '08']#['00', '01', '02', '03', '04', '05', '06', '07', '08']
seqIDtest = ['09', '10']

traintfRecordFLD = "../Data/kitti/train_tfrecords_5tuple/"
testtfRecordFLD = "../Data/kitti/test_tfrecords_5tuple/"

##def main():
#    #find_max_mins("train", pclPath, posePath, seqIDtrain)
#    #find_max_mins("test", pclPath, posePath, seqIDtest)
#    '''
#    We found that the plane slop is in between [ -0.1645 , 0.43 ] # [top, down]
#    So any point beyond this should be trimmed.
#    And all points while converting to depthmap should be grouped in this range for Y
#    Regarding X, we set all points with z > 0. This means slops for X are inf
#
#    We add 2 points to the list holding 2 corners of the image plane
#    normalize points to chunks and then remove the auxiliary points
#    '''
#
#    '''
#    To have all point clouds within same dimensions, we should add extra 0 rows to have them all unified
#    '''
#    #find_max_PCL("train", pclPath, posePath, seqIDtrain)
#    #find_max_PCL("test", pclPath, posePath, seqIDtest)
#
_set_folders(traintfRecordFLD)
_set_folders(testtfRecordFLD)

prepare_dataset("train", pclPath, posePath, seqIDtrain, traintfRecordFLD, numTuples=5)
prepare_dataset("test", pclPath, posePath, seqIDtest, testtfRecordFLD, numTuples=5)
