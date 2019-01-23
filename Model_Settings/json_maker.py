import json
import collections
import numpy as np
import os

def write_json_file(filename, datafile):
    filename = 'Model_Settings/'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent=0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
# Twin Common Parameters
trainLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/train_logs/'
testLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/test_logs/'
warpedTrainDirBase = '../Data/kitti/train_tfrec_itr/'
warpedTestDirBase = '../Data/kitti/test_tfrec_itr/'
####################################################################################
# Data Parameters
numTrainDatasetExamples_desc = "Number of images to process in train dataset"
numTestDatasetExamples_desc = "Number of images to process in test dataset"
trainDataDir_desc = "Directory to read training samples"
testDataDir_desc = "Directory to read test samples"
warpedTrainDataDir_desc = "Directory where to write wrapped train images"
warpedTestDataDir_desc = "Directory where to write wrapped test images"
trainLogDir_desc = "Directory where to write train event logs and checkpoints"
testLogDir_desc = "Directory where to write test event logs and checkpoints"
tMatTrainDir_desc = "tMat Output folder for train"
tMatTestDir_desc = "tMat Output folder for test"
writeWarped_desc = "Flag showing if warped images should be written"
pretrainedModelCheckpointPath_desc = "If specified, restore this pretrained model before beginning any training"
# Image Parameters
imageDepthRows_desc = "Depth Image Height (ROWS)"
imageDepthCols_desc = "Depth Image Width (COLS)"
imageDepthChannels_desc = "Depth Image channels, number of stacked images"
# PCL Parameters
pclRows_desc = "3 rows, xyz of Point Cloud"
pclCols_desc = "Unified number of points (columns) in the Point Cloud"
# tMat Parameters
tMatRows_desc = "rows in transformation matrix"
tMatCols_desc = "cols in transformation matrix"
# Model Parameters
modelName_desc = "Name of the model file to be loaded from Model_Factory"
modelShape_desc = "Network model with 8 convolutional layers with 2 fully connected layers"
numParallelModules_desc = "Number of parallel modules of the network"
batchNorm_desc = "Should we use batch normalization"
weightNorm_desc = "Should we use weight normalization"
optimizer_desc = "Type of optimizer to be used [AdamOptimizer, MomentumOptimizer, GradientDescentOptimizer]"
initialLearningRate_desc = "Initial learning rate."
learningRateDecayFactor_desc = "Learning rate decay factor"
numEpochsPerDecay_desc = "Epochs after which learning rate decays"
momentum_desc = "Momentum Optimizer: momentum"
epsilon_desc = "epsilon value used in AdamOptimizer"
dropOutKeepRate_desc = "Keep rate for drop out"
clipNorm_desc = "Gradient global normalization clip value"
lossFunction_desc = "Indicates type of the loss function to be used [L2, CrossEntropy, ..]"
# Train Parameters
trainBatchSize_desc = "Batch size of input data for train"
testBatchSize_desc = "Batch size of input data for test"
outputSize_desc = "Final output size"
trainMaxSteps_desc = "Number of batches to run"
testMaxSteps_desc = "Number of batches to run during test. numTestDatasetExamples = testMaxSteps x testBatchSize" 
usefp16_desc = "Use 16 bit floating point precision"
logDevicePlacement_desc = "Whether to log device placement"

data = {
    # Data Parameters
    'numTrainDatasetExamples' : 20400,
    'numTestDatasetExamples' : 2790,
    'trainDataDir' : '../Data/kitti/train_tfrec',
    'testDataDir' : '../Data/kitti/test_tfrec',
    'warpedTrainDataDir' : warpedTrainDirBase+'',
    'warpedTestDataDir' : warpedTestDirBase+'',
    'trainLogDir' : trainLogDirBase+'',
    'testLogDir' : testLogDirBase+'',
    'tMatTrainDir' : trainLogDirBase+'/target',
    'tMatTestDir' : testLogDirBase+'/target',
    'writeWarped' : False,
    'pretrainedModelCheckpointPath' : '',
    # Image Parameters
    'imageDepthRows' : 64,#128,
    'imageDepthCols' : 1024,
    'imageDepthChannels' : 2, # All PCL files should have same cols
    # PCL Parameters
    'pclRows' : 3,
    'pclCols' : 62074,
    # tMat Parameters
    'tMatRows' : 3,
    'tMatCols' : 4,
    # Model Parameters
    'modelName' : '',
    'modelShape' : [64, 64, 64, 64, 128, 128, 128, 128, 1024],
    'numParallelModules' : 2,
    'batchNorm' : True,
    'weightNorm' : False,
    'optimizer' : 'MomentumOptimizer', # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
    'initialLearningRate' : 0.005,
    'learningRateDecayFactor' : 0.1,
    'numEpochsPerDecay' : 30000.0,
    'momentum' : 0.9,
    'epsilon' : 0.1,
    'dropOutKeepRate' : 0.5,
    'clipNorm' : 1.0,
    'lossFunction' : 'L2',
    # Train Parameters
    'trainBatchSize' : 16,
    'testBatchSize' : 16,
    'outputSize' : 6, # 6 Params
    'trainMaxSteps' : 90000,
    'testMaxSteps' : 1,
    'usefp16' : False,
    'logDevicePlacement' : False
    }
data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
####################################################################################
####################################################################################
####################################################################################
####################################################################################
##############
reCompileJSON = True
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def write_iterative():
    # Twin Correlation Matching Common Parameters
    trainLogDirBase = '../Data/kitti/logs/tfdh_iterative_logs/train_logs/'
    testLogDirBase = '../Data/kitti/logs/tfdh_iterative_logs/test_logs/'

    data['writeWarpedImages'] = True

    # Iterative model only changes the wayoutput is written, 
    # so any model can be used by ease

    reCompileITR = True
    NOreCompileITR = False
    itr_170523_ITR_B(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_170622_ITR_B(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_170628_ITR_B(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_170706_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_170710_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_170711_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_170719_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_170720_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_170808_ITR_B_inception_n5tuple(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_190101(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_190102(reCompileITR, trainLogDirBase, testLogDirBase)
    itr_190122(reCompileITR, trainLogDirBase, testLogDirBase)
    ##############
    ##############
    ##############

def itr_170523_ITR_B(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'twin_cnn_4p4l2f'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170523_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec'
        data['testDataDir'] = '../Data/kitti/test_tfrec'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170523_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170523_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170523_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)


def itr_170622_ITR_B(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'twin_cnn_4p4l2f'
        data['imageDepthChannels'] = 2
        data['numParallelModules'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170622_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec'
        data['testDataDir'] = '../Data/kitti/test_tfrec'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170622_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170622_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170622_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_170628_ITR_B(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'twin_cnn_6p6l2f'
        data['imageDepthChannels'] = 2
        data['numParallelModules'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170628_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec'
        data['testDataDir'] = '../Data/kitti/test_tfrec'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['modelShape'] = [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170628_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170628_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170628_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
def itr_170706_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170706_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec'
        data['testDataDir'] = '../Data/kitti/test_tfrec'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170706_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170706_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170706_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_170710_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170710_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec'
        data['testDataDir'] = '../Data/kitti/test_tfrec'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170710_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170710_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170710_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [32, 64, 32, 64, 64, 128, 64, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
def itr_170711_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'twin_cnn_4p4l2f_inception'
        data['numParallelModules'] = 3
        data['imageDepthChannels'] = 3
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170711_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec'
        data['testDataDir'] = '../Data/kitti/test_tfrec'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['modelShape'] = [3*8, 3*16, 3*8, 3*16, 64, 128, 64, 128, 1024] 
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170711_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [3*8, 3*16, 3*8, 3*16, 64, 128, 64, 128, 1024] 
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170711_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [3*8, 3*16, 3*8, 3*16, 64, 128, 64, 128, 1024] 
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170711_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [3*8, 3*16, 3*8, 3*16, 64, 128, 64, 128, 1024] 
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
def itr_170719_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        #data['modelName'] = 'twin_cnn_4p4l3f_inception_sepOT'
        data['modelName'] = 'twin_cnn_4p4l3f_inception' # compare loss with 0720
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [2*32, 2*32, 2*32, 2*32, 64, 64, 128, 128, 512, 512]
        data['trainBatchSize'] = 12
        data['testBatchSize'] = 12
        ### ITERATION 1
        runName = '170719_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec'
        data['testDataDir'] = '../Data/kitti/test_tfrec'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170719_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170719_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170719_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
def itr_170720_ITR_B_inception(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'twin_cnn_4p4l3f_inception'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [2*32, 2*32, 2*32, 2*32, 64, 64, 128, 128, 512, 512]
        data['trainBatchSize'] = 12
        data['testBatchSize'] = 12
        ### ITERATION 1
        runName = '170720_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec'
        data['testDataDir'] = '../Data/kitti/test_tfrec'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170720_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170720_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170720_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
def itr_170808_ITR_B_inception_n5tuple(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'twin_cnn_4p4l3f_inception'
        data['numParallelModules'] = 5
        data['imageDepthChannels'] = 5
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [5*16, 32, 5*16, 32, 32, 32, 64, 64, 256, 512]
        data['trainBatchSize'] = 4
        data['testBatchSize'] = 4
        data['numTrainDatasetExamples'] = 20400
        data['numTestDatasetExamples'] = 2790
        ### ITERATION 1
        runName = '170808_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrecords_5tuple'
        data['testDataDir'] = '../Data/kitti/test_tfrecords_5tuple'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170808_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170808_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170808_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_190101(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'cnn_190101'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['trainMaxSteps'] = 150000
        data['lossFunction'] = 'Weighted_Params_L2_l2reg'
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer AdaGrad
        data['initialLearningRate'] = 0.003
        data['learningRateDecayFactor'] = 0.7
        data['outputSize'] = 7 # 4 quat 3 dxyz
        ####################  0    1    2    3 -  4    5    6    7 -   8     9
        data['modelShape'] = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 512]
        data['trainBatchSize'] = 32
        data['testBatchSize'] = 32
        runNameBase = '190101_ITR_B_'
        ### ITERATION 1
        runName = runNameBase+'1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec_'+str(data['numParallelModules'])
        data['testDataDir'] = '../Data/kitti/test_tfrec_'+str(data['numParallelModules'])
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = runNameBase+'2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = runNameBase+'3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = runNameBase+'4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_190102(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'cnn_190102'
        data['numParallelModules'] = 2
        data['imageDepthChannels'] = 2
        data['trainMaxSteps'] = 150000
        data['lossFunction'] = 'Weighted_Params_L2_l2reg'
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer AdaGrad
        #data['initialLearningRate'] = 0.003
        #data['learningRateDecayFactor'] = 0.7 # very fast convergence and then stays almost similar
        data['initialLearningRate'] = 0.001
        data['learningRateDecayFactor'] = 0.7
        data['outputSize'] = 7 # 4 quat 3 dxyz
        ####################  0    1    2    3 -  4    5    6    7 -   8     9
        data['modelShape'] = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 512]
        data['trainBatchSize'] = 32
        data['testBatchSize'] = 32
        runNameBase = '190102_ITR_B_'
        ### ITERATION 1
        runName = runNameBase+'1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec_'+str(data['numParallelModules'])
        data['testDataDir'] = '../Data/kitti/test_tfrec_'+str(data['numParallelModules'])
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = runNameBase+'2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = runNameBase+'3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = runNameBase+'4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def itr_190122(reCompileITR, trainLogDirBase, testLogDirBase):
    if reCompileITR:
        data['modelName'] = 'cnn_190122'
        data['numParallelModules'] = 3
        data['imageDepthChannels'] = 3
        data['trainMaxSteps'] = 150000
        data['lossFunction'] = 'Weighted_Params_3t_L2_l2reg'
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer AdaGrad
        #data['initialLearningRate'] = 0.003
        #data['learningRateDecayFactor'] = 0.7 # very fast convergence and then stays almost similar
        data['initialLearningRate'] = 0.001
        data['learningRateDecayFactor'] = 0.7
        data['outputSize'] = 7 # 4 quat 3 dxyz
        ####################  0    1    2    3 -  4    5    6    7 -   8     9
        data['modelShape'] = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 512]
        data['trainBatchSize'] = 32
        data['testBatchSize'] = 32
        runNameBase = '190102_ITR_B_'
        ### ITERATION 1
        runName = runNameBase+'1'
        data['trainDataDir'] = '../Data/kitti/train_tfrec_'+str(data['numParallelModules'])
        data['testDataDir'] = '../Data/kitti/test_tfrec_'+str(data['numParallelModules'])
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = runNameBase+'2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = runNameBase+'3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = runNameBase+'4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################


def recompile_json_files():
    #write_single()
    #write_twin()
    #write_twin_correlation()
    write_iterative()
    #write_residual()
    print("JSON files updated")
