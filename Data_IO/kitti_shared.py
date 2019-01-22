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
import csv

from pyquaternion import Quaternion

import struct
from scipy import spatial

############################################################################
# xyz[0]/rXYZ out of [-1,1]  this is reveresd
MIN_X_R = -1
MAX_X_R = 1
# xyz[1]/rXYZ out of [-0.12,0.4097]  this is reveresd
MIN_Y_R = -0.12
MAX_Y_R = 0.4097
# Z in range [0.01, 100]
MIN_Z = 0.01
MAX_Z = 100

IMG_ROWS = 64  # makes image of 2x64 = 128
IMG_COLS = 512
PCL_COLS = 62074 # All PCL files should have rows
PCL_ROWS = 3


############################################################################
def _get_tMat_B_2_A(tMatA2o, tMatB2o):
    '''
    3x4 , 3x4 = 3x4
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
    #????????????????????????
    return tMatB2A

def _get_tMat_A_2_B(tMatA2o, tMatB2o):
    '''
    3x4 , 3x4 = 3x4
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

def _get_tMat_B2OvA(tMatA2o, tMatA2B):
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
################# TMAT TO/FROM euler/Quaternions
def _get_quat_dt_from_tmat(tmat):
    '''
    Input:
        tmat is a 3x4 matrix
    Output:
        array of size 7, (4 quaternion [w, i, j, k], 3 deltaT [dx, dy, dz])
    '''
    rotMat = tmat[0:3,0:3] # 3 x 3 rotmat
    deltaT = tmat[2,0:3] # 1 x 3 (dx, dy, dz)
    qt = Quaternion(matrix=rotMat)
    return np.array([qt[0], qt[1], qt[2], qt[3], deltaT[0], deltaT[1], deltaT[2]])

def _get_tmat_from_quat_dt(qtdt):
    '''
    Input:
        array of size 7, (4 quaternion [w, i, j, k], 3 deltaT [dx, dy, dz])
    Output:
        tmat is a 3x4 matrix
    '''
    tmat = np.zeros([3, 4], dtype=np.float32)
    qt = Quaternion([qtdt[0], qtdt[1], qtdt[2], qtdt[3]]) # w, i, j, k
    tmat[:,0:3] = qt.rotation_matrix
    tmat[:,3]=qtdt[4:7] # (dx, dy, dz)
    return tmat

def _get_euler_dt_from_tmat(tmat):
    """
    tmat is a 3x4 matrix
    Output is a 6 valued vector

    For yaw, pitch, roll (alpha, beta, gamma)
    http://planning.cs.uiuc.edu/node103.html
    For dX, dY, dZ
    Last column
    """
    dX = tmat[0][3]
    dY = tmat[1][3]
    dZ = tmat[2][3]
    alpha_yaw = np.arctan2(tmat[1][0], tmat[0][0])
    beta_pitch = np.arctan2(-tmat[2][0], np.sqrt((tmat[2][1]*tmat[2][1])+(tmat[2][2]*tmat[2][2])))
    gamma_roll = np.arctan2(tmat[2][1], tmat[2][2])
    return np.array([alpha_yaw, beta_pitch, gamma_roll, dX, dY, dZ], dtype=np.float32)
    
def _get_tmat_from_euler_dt(abgxyz):
    """
    abgxyz is a 6 valued vector: Alpha_yaw, Beta_pitch, Gamma_roll, DeltaX, DeltaY, DeltaZ
    Output is a 3x4 tmat
    For rotation matrix:
    http://planning.cs.uiuc.edu/node102.html
    For Translation side:
    dX, dY, dZ are last column
    """
    a = abgxyz[0]
    b = abgxyz[1]
    g = abgxyz[2]
    dx = abgxyz[3]
    dy = abgxyz[4]
    dz = abgxyz[5]
    tmat = np.array([
              [np.cos(a)*np.cos(b), (np.cos(a)*np.sin(b)*np.sin(g))-(np.sin(a)*np.cos(g)), (np.cos(a)*np.sin(b)*np.cos(g))+(np.sin(a)*np.sin(g)), dx],
              [np.sin(a)*np.cos(b), (np.sin(a)*np.sin(b)*np.sin(g))+(np.cos(a)*np.cos(g)), (np.sin(a)*np.sin(b)*np.cos(g))-(np.cos(a)*np.sin(g)), dy],
              [-np.sin(b),          np.cos(b)*np.sin(g),                                   np.cos(b)*np.cos(g),                                   dz]
           ], dtype=np.float32)
    return tmat

############################################################################
def get_pose_path(poseFolder, seqID):
    return poseFolder + seqID + ".txt"
def get_pose_data(posePath):
    return np.loadtxt(open(posePath, "r"), delimiter=" ")
############################################################################
def _get_3x4_tmat(poseRow):
    return poseRow.reshape([3,4])
def _add_row4_tmat(pose3x4):
    return np.append(pose3x4, [[0, 0, 0, 1]], axis=0)
def _remove_row4_tmat(pose4x4):
    return np.delete(pose4x4, pose4x4.shape[0]-1, 0)
def get_residual_tMat_A2B(tMatA, tMatB):
    '''
        Input: 3x4, 3x4
        To get residual transformation E:
        T = P x E => (P.inv) x T = (P.inv) x P x E => (P.inv) x T = I x E => (P.inv) x T = E

        return E as residual tMat 3x4
    '''
    # get tMat in the correct form
    tMatA = _add_row4_tmat(_get_3x4_tmat(tMatA))
    tMatB = _add_row4_tmat(_get_3x4_tmat(tMatB))
    tMatResA2B = np.matmul(np.linalg.inv(tMatB), tMatA)
    tMatResA2B = _remove_row4_tmat(tMatResA2B)
    return tMatResA2B

def get_residual_tMat_Bp2B2A(tMatB2A, tMatB2Bp):
    '''
        Input: 3x4, 3x4
        return E as residual tMat 3x4
    '''
    # get tMat in the correct form
    tMatB2A = _add_row4_tmat(_get_3x4_tmat(tMatB2A))
    tMatB2Bp = _add_row4_tmat(_get_3x4_tmat(tMatB2Bp))
    tMatResBp2A = np.matmul(tMatB2A, np.linalg.inv(tMatB2Bp))
    tMatResBp2A = _remove_row4_tmat(tMatResBp2A)
    return tMatResBp2A


################### Training/Testing - Evaluation
def get_training_pose(tmat, mode):
    '''
    Input:
        pose: 3x4
        mode: 'quaternion', 'euler', 'tmat'
    return:
        for 'quaternion' :  7 [w, i, j, k, dX, dY, dZ]
        for 'euler' :       6 [Yaw, Pitch, Roll, dX, dY, dZ]
        for 'tmat' :        12
    '''
    if mode == "quaternion":
        return _get_quat_dt_from_tmat(tmat)
    elif mode == "euler":
        return _get_euler_dt_from_tmat(tmat)
    elif mode == "tmat":
        return tmat
    else:
        raise ValueError("mode should be 'quaternion' or 'euler', not ", mode)
 
def get_training_tmat(pose, mode):
    '''
    Input:
        pose: 
            for 'quaternion' :  7 [w, i, j, k, dX, dY, dZ]
            for 'euler' :       6 [Yaw, Pitch, Roll, dX, dY, dZ]
            for 'tmat' :        12
        mode: 'quaternion', 'euler', 'tmat'
    return:
        tmat: 3x4
    '''
    if mode == "quaternion":
        return _get_tmat_from_quat_dt(pose)
    elif mode == "euler":
        return _get_tmat_from_euler_dt(pose)
    elif mode == "tmat":
        return pose
    else:
        raise ValueError("mode should be 'quaternion' or 'euler', not ", mode)
 

############################################################################
def transform_pcl(xyz, tMat):
    '''
    NEW XYZ = tMAT x XYZ
    pointcloud i 3xN, and tMat 3x4
    '''
    tMat = _add_row4_tmat(_get_3x4_tmat(tMat))
    # append a ones row to xyz
    xyz = np.append(xyz, np.ones(shape=[1, xyz.shape[1]]), axis=0)
    xyz = np.matmul(tMat, xyz)
    # remove last row
    xyz = np.delete(xyz, xyz.shape[0]-1, 0)
    return xyz

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
###############################################
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

############################################################################
def _zero_pad(xyz, num):
    '''
    Append xyz with num 0s to have unified pcl length of PCL_COLS
    '''
    if num < 0:
        print("xyz shape is", xyz.shape)
        print("MAX PCL_COLS is", PCL_COLS)
        raise ValueError('Error... PCL_COLS should be the unified max of the whole system')
    elif num > 0:
        pad = np.zeros([xyz.shape[0], num], dtype=float)
        xyz = np.append(xyz, pad, axis=1)
    # if num is 0 -> do nothing
    return xyz

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

def _add_corner_points(xyz, rXYZ):
    '''
    MOST RECENT CODE A10333
    Add MAX RANGE for xyz[0]/rXYZ out of [-1,1]
    Add MIN RANGE for xyz[1]/rXYZ out of [-0.12,0.4097]
    '''
    ### Add Two corner points with z=0 and x=rand and y calculated based on a, For max min locations
    ### Add Two min-max depth point to correctly normalize distance values
    ### Will be removed after histograms

    xyz = np.append(xyz, [[MIN_Y_R], [MIN_Y_R], [0]], axis=1) # z not needed
    rXYZ = np.append(rXYZ, 1)
    xyz = np.append(xyz, [[MAX_X_R], [MAX_Y_R], [0]], axis=1) # z not needed
    rXYZ = np.append(rXYZ, 1)
    #z = 0.0
    #x = 2.0
    #a = 0.43
    #y = np.sqrt(((a*a)*((x*x)*(x*x)))/(1-(a*a)))
    #xyz = np.append(xyz, [[x], [y], [z]], axis=1)
    #rXYZ = np.append(rXYZ, np.sqrt((x*x)+(y*y)+(z*z)))
    #x = -2.0
    #a = -0.1645
    #y = np.sqrt(((a*a)*((x*x)*(x*x)))/(1-(a*a)))
    #xyz = np.append(xyz, [[x], [y], [z]], axis=1)
    #rXYZ = np.append(rXYZ, np.sqrt((x*x)+(y*y)+(z*z)))

    xyz = np.append(xyz, [[0], [0], [MIN_Z]], axis=1)
    rXYZ = np.append(rXYZ, MIN_Z*MIN_Z)
    xyz = np.append(xyz, [[0], [0], [MAX_Z]], axis=1)
    rXYZ = np.append(rXYZ, MAX_Z*MAX_Z)
    return xyz, rXYZ

def _remove_corner_points(xyz):
    xyz = np.delete(xyz, xyz.shape[1]-1, 1)
    xyz = np.delete(xyz, xyz.shape[1]-1, 1)
    xyz = np.delete(xyz, xyz.shape[1]-1, 1)
    xyz = np.delete(xyz, xyz.shape[1]-1, 1)
    return xyz

def _get_plane_view(xyz, rXYZ):
    ### Flatten to a plane
    # 0 left-right, 1 is up-down, 2 is forward-back
    xT = (xyz[0]/rXYZ).reshape([1, xyz.shape[1]])
    yT = (xyz[1]/rXYZ).reshape([1, xyz.shape[1]])
    zT = rXYZ.reshape([1, xyz.shape[1]])
    planeView = np.append(np.append(xT, yT, axis=0), zT, axis=0)
    return planeView

def _make_image(depthview, rXYZ, **kwargs):
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
    _, xBinEdges = np.histogram(depthview[0], kwargs.get('imageDepthCols'))
    _, yBinEdges = np.histogram(depthview[1], kwargs.get('imageDepthRows'))
    xCent = np.ndarray(shape=xBinEdges.shape[0]-1)
    for i in range(0, xCent.shape[0]):
        xCent[i] = (xBinEdges[i]+xBinEdges[i+1])/2
    yCent = np.ndarray(shape=yBinEdges.shape[0]-1)
    for i in range(0, yCent.shape[0]):
        yCent[i] = (yBinEdges[i]+yBinEdges[i+1])/2
    # make image of size 128x512 : 64 -> 128 (double sampling the height)
    depthImage = np.zeros(shape=[kwargs.get('imageDepthRows'), kwargs.get('imageDepthCols')])
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
        if kwargs.get('imageDepthRows')==128:# hieght is 2x64
            yidx = yidx*2
            depthImage[yidx+1, xidx] = depthview[2, idxs[i]]
        depthImage[yidx, xidx] = depthview[2, idxs[i]]
    return depthImage

def get_depth_image_pano_pclView(xyz, height=1.6, **kwargs):
    '''
    Gets a point cloud
    Keeps points higher than 'height' value and located on the positive Z=0 plane
    Returns correstempMaxponding depthMap and pclView
    '''
    '''
    MOST RECENT CODE A10333
    remove any point who has xyz[0]/rXYZ out of [-1,1]
    remove any point who has xyz[1]/rXYZ out of [-0.12,0.4097]
    '''
    # calc rXYZ
    rXYZ = np.linalg.norm(xyz, axis=0)
    xyz = xyz.transpose()
    first = True
    pclview = np.ndarray(shape=[xyz.shape[1],0], dtype=np.float32)
    for i in range(xyz.shape[0]):
        # xyz[i][2] >= 0 means all the points who have depth larger than 0 (positive depth plane)
        if (xyz[i][2] >= 0) and (xyz[i][1] < height) and (rXYZ[i] > 0) and (xyz[i][0]/rXYZ[i] > -1) and (xyz[i][0]/rXYZ[i] < 1) and (xyz[i][1]/rXYZ[i] > -0.12) and (xyz[i][1]/rXYZ[i] < 0.4097): # frontal view & above ground & x in range & y in range
            if first:
                pclview = xyz[i].reshape(xyz.shape[1], 1)
                first = False
            else:
                pclview = np.append(pclview, xyz[i].reshape(xyz.shape[1], 1), axis=1)
    rPclview = np.linalg.norm(pclview, axis=0)
    depthImage = _make_image(pclview, rPclview, **kwargs)
    pclview = _zero_pad(pclview, PCL_COLS-pclview.shape[1])
    return depthImage, pclview

############################################################################
def remove_trailing_zeros(xyz):
    '''Remove trailing 0 points'''
    condition = (xyz[0] != 0) | (xyz[1] != 0) | (xyz[2] != 0)
    condition = [[condition], [condition], [condition]]
    xyz = np.extract(condition, xyz)
    xyz = xyz.reshape([3, -1])
    return xyz
