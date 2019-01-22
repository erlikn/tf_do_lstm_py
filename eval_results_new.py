from datetime import datetime
import os.path
import time
import json
import importlib
from os import listdir                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
from os.path import isfile, join
print(os.getcwd())
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import Data_IO.kitti_shared as kitti


ID_SEQ_BASE = 100
ID_SMP_BASE = 100000


# import json_maker, update json files and read requested json file
import Model_Settings.json_maker as json_maker
json_maker.recompile_json_files()

# 170706_ITR_B_1.json Best performer
#jsonsToRead = ['170706_ITR_B_1.json',
#               '170706_ITR_B_2.json',
#               '170706_ITR_B_3.json'
#               ]

# BAD zigzagy as the orientation and translation are seperate
#jsonsToRead = ['170720_ITR_B_1.json',
#               '170720_ITR_B_2.json'
#               ]

#jsonsToRead = ['170719_ITR_B_1.json',
#               '170719_ITR_B_2.json'
#               ]

#jsonsToRead = ['190101_ITR_B_1.json']
jsonsToRead = ['190102_ITR_B_1.json']
POSE_MODE = 'quaternion' # 'quaternion', 'euler', 'tmat'

############# SET PRINT PRECISION
np.set_printoptions(precision=4, suppress=True)
############# STATE
PHASE = 'train' # 'train' or 'test'
############# PATHS
pclPath = '../Data/kitti/pointcloud/'
posePath = '../Data/kitti/poses/'
seqIDtrain = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
seqIDtest = ['09', '10']

###########################################################
def _GetControlParams(modelParams):
    """
    Get control parameters for the specific task
    """
    modelParams['phase'] = PHASE
    #params['shardMeta'] = model_cnn.getShardsMetaInfo(FLAGS.dataDir, params['phase'])
    modelParams['existingParams'] = None
    modelParams['gTruthDir'] = posePath
    if modelParams['phase'] == 'train':
        modelParams['activeBatchSize'] = modelParams['trainBatchSize']
        modelParams['maxSteps'] = modelParams['trainMaxSteps']
        modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
        modelParams['dataDir'] = modelParams['trainDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTrainDataDir']
        modelParams['tMatDir'] = modelParams['tMatTrainDir']
        modelParams['seqIDs'] = seqIDtrain
    if modelParams['phase'] == 'test':
        modelParams['activeBatchSize'] = modelParams['testBatchSize']
        modelParams['maxSteps'] = modelParams['testMaxSteps']
        modelParams['numExamples'] = modelParams['numTestDatasetExamples']
        modelParams['dataDir'] = modelParams['testDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTestDataDir']
        modelParams['tMatDir'] = modelParams['tMatTestDir']
        modelParams['seqIDs'] = seqIDtest
    return modelParams

def read_model_params(jsonToRead):
    print("Reading %s" % jsonToRead)
    with open('Model_Settings/'+jsonToRead) as data_file:
        modelParams = json.load(data_file)
    _GetControlParams(modelParams)
    print('Evaluation phase for %s' % modelParams['phase'])
    print('Ground truth input: %s' % modelParams['gTruthDir'])
    if modelParams['phase'] == 'train':
        print('Train sequences:', seqIDtrain)
        print('Prediction input: %s' % modelParams['tMatDir'])
    else:
        print('Test sequences:' % seqIDtest)
        print('Prediction Input: %s' % modelParams['tMatDir'])
    print(modelParams['modelName'])
    #if input("IS PRESENTED INFORMATION VALID? ") != "yes":
    #    print("Please consider updating the provided information!")
    #    return
    return modelParams

################################################################### Visualization functions
def VisalizePath(xyzlist, legendNames):
    import matplotlib.pyplot as plt
    colors = ['r', 'b', 'c', 'y', 'k', 'm', 'p']
    pltlist = list()
    for i in range(len(xyzlist)):
        print('     plotting points = ', (xyzlist[i].shape))
        pltmap, = plt.plot(xyzlist[i][0], xyzlist[i][1], colors[i])
        pltlist.append(pltmap)
    #pred2, = plt.plot(p2xyz[0], p2xyz[1], 'c', alpha=0.5)
    plt.legend(pltlist, legendNames)
    plt.show()
    #plt.draw()
    #plt.waitforbuttonpress()
    #plt.close()

def VisalizePath3D(xyzlist, legendNames):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    colors = ['r', 'b', 'c', 'y', 'k', 'm', 'p']
    pltlist = list()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(xyzlist)):
        print('     plotting points = ', (xyzlist[i].shape))
        pltmap = ax.scatter(xyzlist[i][0], xyzlist[i][1], xyzlist[i][2], c=colors[i])        
        #pltmap, = plt.scatter(xyzlist[i][0], xyzlist[i][1], xyzlist[i][2], colors[i])
        pltlist.append(pltmap)
    #pred2, = plt.plot(p2xyz[0], p2xyz[1], 'c', alpha=0.5)
    plt.legend(pltlist, legendNames)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    #plt.draw()
    #plt.waitforbuttonpress()
    #plt.close()

################################################################### Input functions
def _GetFileNames(readFolder, fileFormat):
    print(readFolder, '---', fileFormat)
    filenames = [f for f in listdir(readFolder) if (isfile(join(readFolder, f)) and fileFormat in f)]
    return filenames

def GetPredParamsFile(readFolder, fileFormat):
    """
    read all predictions of all sequences to a list
    """
    filenames = _GetFileNames(readFolder, fileFormat)
    predAllList = list()
    predAllListTemp = list()
    for i in range(9):
        predAllListTemp.append(list())
    for i in range(0,len(filenames)):
        with open(readFolder+'/'+filenames[i]) as data_file:
            tMatJson = json.load(data_file)
            #print(tMatJson)
        predAllListTemp[int(tMatJson['seq']-ID_SEQ_BASE)].append(tMatJson)
    for i in range(9):       
        seqList = sorted(predAllListTemp[i], key=lambda k: k['idx'])
        predAllList.append(seqList)
    return predAllList

def import_gt_data(model_params):
    gt_pose = list()
    gt_tmat = list()
    for i in range(len(model_params['seqIDs'])):
        print("Collecting GT Data: {0} / {1}".format(i+1, len(model_params['seqIDs'])))
        ###### Create GT map
        gt_pose_id, gt_tmat_id = GetGtPoseTmat(model_params['gTruthDir'], model_params['seqIDs'][i])
        gt_pose.append(gt_pose_id)
        gt_tmat.append(gt_tmat_id)
        #gt_map_pose = GetMapViaPose(gt_pose) # PoseX2O
        #gt_map_tmat = GetMapViaTmat(gt_tmat) # tmatB2A
        #VisalizePath([gt_map_pose, gt_map_tmat], ['GT_Pose', 'GT_Tmat'])
    gt_data={'pose': gt_pose, 'tmat': gt_tmat}
    return gt_data
################################################################### MAP FUNCTIONS
def GetMapViaPose(poseX2Olist):
    """
    Input:
        poselist X2O
    Output:
        Map
    """
    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    pathMap = np.append(pathMap, origin, axis=1)
    for i in range(len(poseX2Olist)):
        pointT = kitti.transform_pcl(origin, poseX2Olist[i])
        pathMap = np.append(pathMap, pointT, axis=1)
    return pathMap

def GetMapViaTmat(tmatlist):
    """
    Input:
        tmatList B2A
    Output:
        Map
    """
    '''
    Sequencing : P(n) -> P(n-1) -> ... -> P(O)
    '''
    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    pathMap = np.append(pathMap, origin, axis=1)
    for i in range(len(tmatlist)-1,-1,-1): #last index to 0
        pathMap = kitti.transform_pcl(pathMap, tmatlist[i])
        pathMap = np.append(pathMap, origin, axis=1)
    #pathMap = np.flip(pathMap)
    return pathMap
################################################################### Pose Tmat Functions
def GetGtPoseTmat(gtPoseDir, seqID):
    '''
    '''
    gtPosePath = kitti.get_pose_path(gtPoseDir, seqID)
    gtPoseList = kitti.get_pose_data(gtPosePath) # gtpose -> X2O (12 values)
    gtPose = list()
    for i in range(0, len(gtPoseList)):
        gtPose.append(kitti._get_3x4_tmat(gtPoseList[i]))
    #### gtpose -> prediction like
    gtTmat = list()
    # ntuple == 2, so I am appending the first X2O and the rest are B2A
    gtTmat.append(gtPose[0])
    for i in range(1, len(gtPose)):
        gtTmat.append(kitti._get_tMat_B_2_A(gtPose[i-1], gtPose[i]))
    ####
    return gtPose, gtTmat

def GetPredTmat(p_params_transi, gt_nt_init, mode, p_tmat_prev=list()):
    '''
    Total transition Final B->A = B->A' --> A'->A" --> A"->A
    Input:
        p_params_transi = parametric transitional predictions (A"->A) 
        gt_nt_init = ground truth tmat data for initial (ntuple-1)
        mode = 'target', 'pred'
        p_tmat_prev = prediction total to previous stage (B->A" = T[A'->A"]*T[B->A'])
    Output:
        p_tmat B->A = T[A"->A]*p_tmat_prev
    '''
    #### get prediction for an specific sequence
    print("Pred params count:", len(p_params_transi))
    p_params = list()
    idx_curr = p_params_transi[0]['idx']
    for i in range(len(p_params_transi)):
        #### test the prediction
        if mode == 'pred':
            p_params.append(p_params_transi[i]['pred'])
        #### test the GT
        if mode == 'target':
            if POSE_MODE == 'quaternion':
                p_params.append(np.reshape(p_params_transi[i]['target'], 7))
            elif POSE_MODE == 'euler':
                p_params.append(np.reshape(p_params_transi[i]['target'], 6))
            elif POSE_MODE == 'tmat':
                p_params.append(np.reshape(p_params_transi[i]['target'], 12))
        if idx_curr != p_params_transi[i]['idx']:
            raise ValueError("Predictions are not sequential... index ", i)
        idx_curr += 1
    p_params = np.array(p_params)
    #### get transitional tmat
    p_tmat_transi = gt_nt_init
    for i in range(p_params.shape[0]):
        p_tmat_transi.append(kitti.get_training_tmat(p_params[i], POSE_MODE))
    #### get final tmat
    if len(p_tmat_prev) > 0:
        # get p_tmat as p_tmat_transi*p_tmat_prev
        raise ValueError("Not developed yet!")
    else:
        p_tmat = p_tmat_transi
    return p_tmat
################################################################### Evaluation

def evaluate(model_params, gt_data, p_tmat_prev=list()):
    # For each sequence
    # Read all prediction posefiles and sort them based on the seqID and frameID
    p_par_transi = GetPredParamsFile(model_params['tMatDir'], ".json")
    print("Prediction parameters for all sequences are loaded")
    for i in range(len(model_params['seqIDs'])):
        print("Processing sequences: {0} / {1}".format(i+1, len(model_params['seqIDs'])))
        ###### Create GT map
        #gt_pose = gt_data['pose'][i]
        #print("GT pose count:", len(gt_pose))
        gt_tmat = gt_data['tmat'][i]
        print("GT tmat B2A count:", len(gt_tmat))
        #gt_map_pose = GetMapViaPose(gt_pose) # PoseX2O
        gt_map_tmat = GetMapViaTmat(gt_tmat) # tmatB2A
        #VisalizePath([gt_map_pose, gt_map_tmat], ['GT_Pose', 'GT_Tmat'])

        # ###### Create P map
        seq_id = int(model_params['seqIDs'][i])
        p_tmat_p = GetPredTmat(p_par_transi[seq_id], gt_tmat[:model_params['numParallelModules']-1], 'pred', p_tmat_prev[seq_id])
        p_tmat_t = GetPredTmat(p_par_transi[seq_id], gt_tmat[:model_params['numParallelModules']-1], 'target', p_tmat_prev[seq_id])
        print("Pred tmat P B2A count:", len(p_tmat_p))
        print("Pred tmat T B2A count:", len(p_tmat_t))
        p_map_tmat_p = GetMapViaTmat(p_tmat_p)
        p_map_tmat_t = GetMapViaTmat(p_tmat_t)
        #VisalizePath([gt_map_tmat, p_map_tmat], ['GT_tmat', 'P_tmat'])
        #VisalizePath3D([gt_map_tmat, p_map_tmat], ['GT_tmat', 'P_tmat'])
        VisalizePath3D([p_map_tmat_t, p_map_tmat_p], ['P_tmat_T', 'P_tmat_P'])
        p_tmat_prev[seq_id] = p_map_tmat_p

        # ######
        # # difference analysis
        # print(len(pPoseParam), len(gtParam))
        # print("Abs error:", np.sum(np.abs(pParam-np.array(gtParam)), axis=0))
        # print("+/- error:", np.sum((pParam-np.array(gtParam)), axis=0))

    return p_tmat_prev

def main(argv=None):  # pylint: disable=unused-argumDt
    model_params = read_model_params(jsonsToRead[0])
    gt_data = import_gt_data(model_params)
    p_tmat_prev = [[],[],[],[],[],[],[],[],[],[]]
    p_tmat_prev = evaluate(model_params, gt_data, p_tmat_prev)
    if (len(jsonsToRead)>1):
        model_params = read_model_params(jsonsToRead[1])
        p_tmat_prev = evaluate(model_params, gt_data, p_tmat_prev)
    if (len(jsonsToRead)>2):
        model_params = read_model_params(jsonsToRead[2])
        p_tmat_prev = evaluate(model_params, gt_data, p_tmat_prev)
    if (len(jsonsToRead)>3):
        model_params = read_model_params(jsonsToRead[3])
        p_tmat_prev = evaluate(model_params, gt_data, p_tmat_prev)
main()
