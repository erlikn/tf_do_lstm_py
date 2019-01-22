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

"""Builds the calusa_heatmap network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

import Model_Factory.model_base as model_base

USE_FP_16 = False

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def inference_l2reg(images, **kwargs): #batchSize=None, phase='train', outLayer=[13,13], existingParams=[]
    modelShape = kwargs.get('modelShape')
    wd = None #0.0002
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    batchSize = kwargs.get('activeBatchSize', None)
    l2reg = 0
    #######################################
    ############# Parallel Units
    print("parallel units")
    ############# CONV_P0  -  1024 x 64
    fireOut, prevExpandDim, l2regLayer = model_base.conv_fire_parallel_module_l2regul('conv_p0', images, kwargs.get('imageDepthChannels'),
                                                                  {'cnn3x3': modelShape[0]},
                                                                  stride=[1, 2, 2, 1],
                                                                  wd=wd, **kwargs)
    l2reg += l2regLayer
    ############# CONV_P1  -  512 x 32
    fireOut, prevExpandDim, l2regLayer = model_base.conv_fire_parallel_module_l2regul('conv_p1', fireOut, prevExpandDim,
                                                                  {'cnn3x3': modelShape[1]},
                                                                  stride=[1, 2, 2, 1],
                                                                  wd=wd, **kwargs)
    l2reg += l2regLayer
    ############# CONV_P2  -  256 x 16
    fireOut, prevExpandDim, l2regLayer = model_base.conv_fire_parallel_module_l2regul('conv_p2', fireOut, prevExpandDim,
                                                                  {'cnn3x3': modelShape[2]},
                                                                  stride=[1, 2, 2, 1],
                                                                  wd=wd, **kwargs)
    l2reg += l2regLayer
    ############# CONV_P3  -  128 x 8
    fireOut, prevExpandDim, l2regLayer = model_base.conv_fire_parallel_module_l2regul('conv_p3', fireOut, prevExpandDim,
                                                                  {'cnn3x3': modelShape[3]},
                                                                  stride=[1, 2, 2, 1],
                                                                  wd=wd, **kwargs)
    l2reg += l2regLayer
    #######################################
    ####################################### Matching Units
    print("matching units")
    ############# CONV_M4  -  64 x 4
    fireOut_m4, prevExpandDim_m4, l2regLayer = model_base.conv_fire_module_l2regul('conv_m4', fireOut, prevExpandDim,
                                                                  {'cnn1x1': modelShape[4]},
                                                                  wd=wd, 
                                                                  stride=[1, 1, 1, 1],
                                                                  **kwargs)
    l2reg += l2regLayer
    ############# CONV_M5  -  64 x 4
    fireOut, prevExpandDim, l2regLayer = model_base.conv_fire_module_l2regul('conv_m5', fireOut_m4, prevExpandDim_m4,
                                                                  {'cnn1x3': modelShape[5]},
                                                                  wd=wd, 
                                                                  stride=[1, 1, 2, 1],
                                                                  **kwargs)
    l2reg += l2regLayer
    ############# CONV_M6  -  32 x 4
    fireOut, prevExpandDim, l2regLayer = model_base.conv_fire_module_l2regul('conv_m6', fireOut, prevExpandDim,
                                                                  {'cnn1x1': modelShape[6]},
                                                                  wd=wd, 
                                                                  stride=[1, 1, 1, 1],
                                                                  **kwargs)
    l2reg += l2regLayer
    ############# CONV_M7  -  16 x 4
    fireOut, prevExpandDim, l2regLayer = model_base.conv_fire_module_l2regul('conv_m7', fireOut, prevExpandDim,
                                                                  {'cnn1x3': modelShape[7]},
                                                                  wd=wd, 
                                                                  stride=[1, 1, 2, 1],
                                                                  **kwargs)
    l2reg += l2regLayer
    #######################################
    ############# Concat before INC
    fireOut_m4 = tf.nn.max_pool(fireOut_m4, [1,1,4,1], [1,1,4,1], 'SAME')
    fireOut = tf.concat([fireOut, fireOut_m4], axis=3)
    prevExpandDim = prevExpandDim+prevExpandDim_m4
    #######################################
    ############# Dropout before INC
    with tf.name_scope("drop_inc"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
        fireOut = tf.nn.dropout(fireOut, keepProb, name="dropout_inc")
    #######################################
    print("inception units")
    ############# Inception to get FC
    ############# CONV_INC1 - 16 x 4
    fireOut, prevExpandDim, l2regLayer = model_base.conv_fire_inception_module_l2reg('conv_inc1', fireOut, prevExpandDim,
                                                       {'cnnFC': modelShape[8]},
                                                       wd, **kwargs)
    l2reg += l2regLayer
    ############# CONV_M8  -  4 x 2
    fireOut, prevExpandDim, l2regLayer = model_base.conv_fire_inception_module_l2reg('conv_inc2', fireOut, prevExpandDim,
                                                       {'cnnFC': modelShape[8]},
                                                       wd, **kwargs)
    l2reg += l2regLayer
    #######################################
    ############# Linearization assertion  -  [batchsize, 1, 1, prevExpandDim]
    fireOut = tf.reshape(fireOut, [batchSize, prevExpandDim])
    #######################################
    ############# Dropout before FC
    with tf.name_scope("drop_fc"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
        fireOut = tf.nn.dropout(fireOut, keepProb, name="dropout_fc")
    #######################################
    print("fc units")
    ############# Fully Connected Layers 
    ############# FC1 
    fireOut, prevExpandDim, l2regLayer = model_base.fc_fire_module_l2regul('fc1', fireOut, prevExpandDim,
                                                             {'fc': modelShape[9]},
                                                             wd, **kwargs)
    l2reg += l2regLayer
    ############# FC2 - output layer
    fireOut, prevExpandDim, l2regLayer = model_base.fc_fire_module_l2regul('fc2_output', fireOut, prevExpandDim,
                                                             {'fc': kwargs.get('outputSize')},
                                                             wd, **kwargs)
    l2reg += l2regLayer
    #######################################
    return fireOut, l2reg

def loss(pred, target, **kwargs): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    Returns:
      Loss tensor of type float.
    """
    return model_base.loss(pred, target, **kwargs)

def loss_l2reg(pred, target, l2reg, **kwargs): # batchSize=Sne
    return model_base.loss_l2reg(pred, target, l2reg, **kwargs)

def train(loss, globalStep, **kwargs):
    return model_base.train(loss, globalStep, **kwargs)

def test(loss, globalStep, **kwargs):
    return model_base.test(loss, globalStep, **kwargs)
