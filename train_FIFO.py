# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division

from datetime import datetime
import os, os.path
import time
import logging
import json
import importlib

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.python.debug as tf_debug

PHASE = 'train'
# import json_maker, update json files and read requested json file
import Model_Settings.json_maker as json_maker
json_maker.recompile_json_files()
jsonToRead = '170719_ITR_B_1.json'
print("Reading %s" % jsonToRead)
with open('Model_Settings/'+jsonToRead) as data_file:
    modelParams = json.load(data_file)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

# import input & output modules 
import Data_IO.data_input as data_input
import Data_IO.data_output as data_output

# import corresponding model name as model_cnn, specifed at json file
model_cnn = importlib.import_module('Model_Factory.'+modelParams['modelName'])

####################################################
####################################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('printOutStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('summaryWriteStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('modelCheckpointStep', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportStep', 250,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportOutputWrite', 250,
                            """Number of batches to run.""")
####################################################
####################################################
def _get_control_params():
    modelParams['phase'] = PHASE
    #params['shardMeta'] = model_cnn.getShardsMetaInfo(FLAGS.dataDir, params['phase'])

    modelParams['existingParams'] = None

    if modelParams['phase'] == 'train':
        modelParams['activeBatchSize'] = modelParams['trainBatchSize']
        modelParams['maxSteps'] = modelParams['trainMaxSteps']
        modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
        modelParams['dataDir'] = modelParams['trainDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTrainDataDir']

    if modelParams['phase'] == 'test':
        modelParams['activeBatchSize'] = modelParams['testBatchSize']
        modelParams['maxSteps'] = modelParams['testMaxSteps']
        modelParams['numExamples'] = modelParams['numTestDatasetExamples']
        modelParams['dataDir'] = modelParams['testDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTestDataDir']

####################################################
####################################################
def train():
    _get_control_params()

    with tf.Graph().as_default():
        # track the number of train calls (basically number of batches processed)
        globalStep = tf.get_variable('globalStep',
                                     [],
                                     initializer=tf.constant_initializer(0),
                                     trainable=False)

        # Get images and transformation for model_cnn.
        images, pclA, pclB, targetT, tfrecFileIDs = data_input.inputs(**modelParams)
        print('Input        ready')
        # Build a Graph that computes the HAB predictions from the
        # inference model.
        # Build an initialization operation to run below.
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        config = tf.ConfigProto(log_device_placement=modelParams['logDevicePlacement'])
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)
        print('Session      ready')

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)


        # restore a saver.
        #saver.restore(sess, (modelParams['trainLogDir'].replace('_B_2','_B_1'))+'/model.ckpt-'+str(modelParams['trainMaxSteps']-1))
        #print('Ex-Model     loaded')

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        print('QueueRunner  started')

        
        print('Training     started')
        durationSum = 0
        durationSumAll = 0
        for step in xrange(20):
            startTime = time.time()
            evalTfrecFileIDs = sess.run(tfrecFileIDs)
            duration = time.time() - startTime
            durationSum += duration
            print(evalTfrecFileIDs)
        ######### USE LATEST STATE TO WARP IMAGES
        ### outputDIR = modelParams['warpedOutputFolder']+'/'
        ### outputDirFileNum = len([name for name in os.listdir(outputDIR) if os.path.isfile(os.path.join(outputDIR, name))])
### 
### 
        ### durationSum = 0
        ### durationSumAll = 0
        ### if modelParams['writeWarpedImages']:
        ###     lossValueSum = 0
        ###     stepsForOneDataRound = int((modelParams['numExamples']/modelParams['activeBatchSize']))
        ###     print('Warping %d images with batch size %d in %d steps' % (modelParams['numExamples'], modelParams['activeBatchSize'], stepsForOneDataRound))
        ###     #for step in xrange(stepsForOneDataRound):
        ###     step = 0
        ###     while outputDirFileNum != 20400:
        ###         startTime = time.time()
        ###         evImages, evPclA, evPclB, evtargetT, evtargetP, evtfrecFileIDs, evlossValue = sess.run([images, pclA, pclB, targetT, targetP, tfrecFileIDs, loss])
        ###         #### put imageA, warpped imageB by pHAB, HAB-pHAB as new HAB, changed fileaddress tfrecFileIDs
        ###         data_output.output(evImages, evPclA, evPclB, evtargetT, evtargetP, evtfrecFileIDs, **modelParams)
        ###         duration = time.time() - startTime
        ###         durationSum += duration
        ###         durationSumAll += duration
        ###         # Print Progress Info
        ###         if ((step % FLAGS.ProgressStepReportOutputWrite) == 0) or ((step+1) == stepsForOneDataRound):
        ###             print('Progress: %.2f%%, Loss: %.2f, Elapsed: %.2f mins, Training Completion in: %.2f mins' % 
        ###                     ((100*step)/stepsForOneDataRound, evlossValue/(step+1), durationSum/60, (((durationSum*stepsForOneDataRound)/(step+1))/60)-(durationSum/60)))
        ###             #print('Total Elapsed: %.2f mins, Training Completion in: %.2f mins' % 
        ###             #        durationSumAll/60, (((durationSumAll*stepsForOneDataRound)/(step+1))/60)-(durationSumAll/60))
        ###         outputDirFileNum = len([name for name in os.listdir(outputDIR) if os.path.isfile(os.path.join(outputDIR, name))])
        ###         step+=1
        ###     print('Average training loss = %.2f - Average time per sample= %.2f s, Steps = %d' % (evlossValue/modelParams['activeBatchSize'], durationSum/(step*modelParams['activeBatchSize']), step))

def main(argv=None):  # pylint: disable=unused-argumDt
    print(modelParams['modelName'])
    print('Rounds on datase = %.1f' % float((modelParams['trainBatchSize']*modelParams['trainMaxSteps'])/modelParams['numTrainDatasetExamples']))
    print('Train Input: %s' % modelParams['trainDataDir'])
    #print('Test  Input: %s' % modelParams['testDataDir'])
    print('Train Logs Output: %s' % modelParams['trainLogDir'])
    #print('Test  Logs Output: %s' % modelParams['testLogDir'])
    print('Train Warp Output: %s' % modelParams['warpedTrainDataDir'])
    #print('Test  Warp Output: %s' % modelParams['warpedTestDataDir'])
    print('')
    print('')
    train()


if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()
