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

import re

import tensorflow as tf
import numpy as np

import Model_Factory.optimizer_params as optimizer_params
import Model_Factory.loss_base as loss_base

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
    """Helper to create summaries for activations.
    
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    #tf.histogram_summary(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer, dtype):
    """Helper to create a Variable stored on CPU memory.
    
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, initializer, dtype, wd, trainable=True):
    """Helper to create an initialized Variable with weight decay.
    
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    
    Returns:
      Variable Tensor
    """     
    # tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    if isinstance(initializer, np.ndarray):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            var = tf.get_variable(name, 
                                  initializer=initializer,
                                  dtype=dtype,
                                  trainable=trainable)
    else:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            var = tf.get_variable(name, 
                                  shape, 
                                  initializer=initializer, 
                                  dtype=dtype, 
                                  trainable=trainable)
    #if wd is not None:
    #    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    #    tf.add_to_collection('losses', weight_decay)
        
    return var

def batch_norm(name, tensorConv, dtype, phase):
    if phase == 'train':
        training = True
        print('         batchNorm train')
        #virtualBatchSize = None
    else:
        training = False
        print('         batchNorm TEST')
    return tf.layers.batch_normalization(tensorConv, name=name, training=training)
    #return tensorConv

###############################################################    

def twin_correlation(name, prevLayerOut, prevLayerDim, d, s2, **kwargs):
    dtype = tf.float16 if kwargs.get('usefp16') else tf.float32

    D = (2*d)+1
    numParallelModules = kwargs.get('numParallelModules') # 2
    # Input from Twin network -> numParallelModules = 2
    # Split tensor through last dimension into numParallelModules tensors
    prevLayerOut = tf.split(prevLayerOut, num_or_size_splits=numParallelModules, axis=3)
    prevLayerIndivDims = prevLayerDim / numParallelModules

    # f1 is a list of size batchSize,x,y of [1,1,c,1] tensors  => last 1 is for the output size
    prevLayerOut[0] = tf.split(prevLayerOut[0], num_or_size_splits=kwargs.get('activeBatchSize'), axis=0) # batches seperate

    for i in range(len(prevLayerOut[0])):
        prevLayerOut[0][i] = tf.split(prevLayerOut[0][i], prevLayerOut[0][i].get_shape()[0], axis=0) # X seperate
        for j in range(len(prevLayerOut[0][i])):
            prevLayerOut[0][i][j] = tf.split(prevLayerOut[0][i][j], prevLayerOut[0][i][j].get_shape()[1], axis=1) # Y seperate
            for k in range(len(prevLayerOut[0][i][j])):
                prevLayerOut[0][i][j][k] = tf.reshape(prevLayerOut[0][i][j][k], [1, 1, int(prevLayerOut[0][i][j][k].get_shape()[2]), 1])
                prevLayerOut[0][i][j][k].set_shape([1, 1, int(prevLayerOut[0][i][j][k].get_shape()[2]), 1])

    # padd f2 with 20 zeros on each side
    # output is batch x [1, x+2d, y+2d, c]
    prevLayerOut[1] = tf.split(prevLayerOut[1], prevLayerOut[1].get_shape()[0], axis=0) # batches seperate
    for i in range(len(prevLayerOut[1])):
        prevLayerOut[1][i] = tf.pad(prevLayerOut[1][i], [[0,0],[d,d],[d,d],[0,0]], mode='CONSTANT', name=None)

    # correlation
    with tf.variable_scope(name):
        with tf.variable_scope('corr1x1') as scope:
            for batches in range(len(prevLayerOut[0])): # batches
                for i in range(len(prevLayerOut[0][batches])): # X
                    for j in range(len(prevLayerOut[0][batches][i])): # Y
                        # kernel_shape = width, height, dims (input), dims (output) [1,1,c,1]
                        f1kernel = prevLayerOut[0][batches][i][j]
                        ##print(f1kernel.get_shape())
                        # inputs of [1, x+2d, y+2d, c]
                        # Select square subtensor centered at i,j with square size D=2*d+1 and stride 2 in each axis with complete channel data
                        # if d=20, s2=2 => f2box = [1,21,21,1]
                        f2box = tf.strided_slice(prevLayerOut[1][batches], [0, i, j], [1, i+D, j+D], [1, s2, s2])
                        ##print(f2box.get_shape())
                        # corrsingle is a [1,21,21,1]
                        # reshape to [1,1,1,21*21] or [1,1,1,441]
                        corrSingle = tf.reshape(tf.nn.conv2d(f2box, f1kernel, [1, 1, 1, 1], padding='SAME'), [1, 1, 1, (d+1)*(d+1)])
                        # concat along height dimension
                        if j == 0:
                            corrBox = corrSingle
                        else:
                            corrBox = tf.concat([corrBox, corrSingle], axis=2)
                    # concat along width dimension
                    if i == 0:
                        resultTensor = corrBox
                    else:
                        resultTensor = tf.concat([resultTensor, corrBox], axis=1)
                # concat along batches
                if batches == 0:
                    resultBatch = resultTensor
                else:
                    resultBatch = tf.concat([resultBatch, resultTensor], axis=0)
    #return resultTensor
    return resultBatch, D*D

def twin_correlation_simple(name, prevLayerOut, prevLayerDim, **kwargs):
    dtype = tf.float16 if kwargs.get('usefp16') else tf.float32

    numParallelModules = kwargs.get('numParallelModules') # 2
    # Input from Twin network -> numParallelModules = 2
    # Split tensor through last dimension into numParallelModules tensors
    prevLayerOut = tf.split(prevLayerOut, num_or_size_splits=numParallelModules, axis=3)
    prevLayerIndivDims = prevLayerDim / numParallelModules

    # f1 is a list of size batchSize,x,y of [1,1,c,1] tensors  => last 1 is for the output size
    prevLayerOut[0] = tf.split(prevLayerOut[0], num_or_size_splits=kwargs.get('activeBatchSize'), axis=0) # batches seperate

    #ins = tf.split(input, num_or_size_splits=32, axis=0)
    # TO BE COMPLETED

    resultBatch = tf.reshape(tf.nn.conv2d(prevLayerOut[0], prevLayerOut[1], [1, 4, 4, 1], padding='SAME'), [kwargs.get('activeBatchSize'), -1])

    #return resultTensor
    return resultBatch, 

def conv_fire_parallel_residual_module(name, prevLayerOut, prevLayerDim, historicLayerOut, historicLayerDim, fireDimsSingleModule, numParallelModules, wd=None, **kwargs):
    """
    Input Args:
        name:               scope name
        prevLayerOut:       output tensor of previous layer
        prevLayerDim:      size of the last (3rd) dimension in prevLayerOut
        numParallelModules: number of parallel modules and parallel data in prevLayerOut
        fireDimsSingleModule:     number of output dimensions for each parallel module
    """
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')
    
    # Twin network -> numParallelModules = 2
    # Split tensor through last dimension into numParallelModules tensors
    prevLayerOut = tf.split(prevLayerOut, numParallelModules, axis=3)
    prevLayerIndivDims = prevLayerDim / numParallelModules

    historicLayerOut = tf.split(historicLayerOut, numParallelModules, axis=3)
    historicLayerIndivDim = historicLayerDim / numParallelModules

    if (fireDimsSingleModule.get('cnn1x1')):
        cnnName = 'cnn1x1'
        kernelSizeR = 1
    if (fireDimsSingleModule.get('cnn3x3')):
        cnnName = 'cnn3x3'
        kernelSizeR = 3
    if (fireDimsSingleModule.get('cnn5x5')):
        cnnName = 'cnn5x5'
        kernelSizeR = 5
    if (fireDimsSingleModule.get('cnn7x7')):
        cnnName = 'cnn7x7'
        kernelSizeR = 7

    # output depth of the convolution should be same as historic
    if numParallelModules*fireDimsSingleModule[cnnName] != historicLayerDim:
        # TO DO
        if kwargs.get('residualPadding') == "conv":
            for prl in range(numParallelModules):
                # convlove historic data with a kernel of 1x1 and output size of prevLayerDim
                historicLayerOut[prl] = historicLayerOut[prl]
        if kwargs.get('residualPadding') == "zeroPad":
            for prl in range(numParallelModules):
                # zero pad current historicLayerOut to the size of prevLayerDim
                historicLayerOut[prl] = historicLayerOut[prl]


    with tf.variable_scope(name):
        with tf.variable_scope(cnnName) as scope:
            layerName = scope.name.replace("/", "_")
            #kernel = _variable_with_weight_decay('weights',
            #                                     shape=[kernelSizeR, kernelSizeR, prevLayerIndivDims, fireDimsSingleModule['cnn3x3']],
            #                                     initializer=existingParams[layerName]['weights'] if (existingParams is not None and
            #                                                                                         layerName in existingParams) else
            #                                                    (tf.contrib.layers.xavier_initializer_conv2d() if kwargs.get('phase')=='train' else
            #                                                     tf.constant_initializer(0.0, dtype=dtype)),
            #                                     dtype=dtype,
            # wd=wd,
            #                                     trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
            #                                                                               layerName in existingParams) else True)
            stddev = np.sqrt(2/np.prod(prevLayerOut[0].get_shape().as_list()[1:]))
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[kernelSizeR, kernelSizeR, prevLayerIndivDims, fireDimsSingleModule[cnnName]],
                                                 initializer=existingParams[layerName]['weights'] if (existingParams is not None and
                                                                                                     layerName in existingParams) else
                                                                (tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train' else
                                                                 tf.constant_initializer(0.0, dtype=dtype)),
                                                 dtype=dtype,
                                                 wd=wd,
                                                 trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            if existingParams is not None and layerName in existingParams:
                biases = tf.get_variable('biases',
                                         initializer=existingParams[layerName]['biases'], 
                                         dtype=dtype)
            else:
                biases = tf.get_variable('biases', [fireDimsSingleModule[cnnName]],
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         dtype=dtype)

            for prl in range(numParallelModules):
                conv = tf.nn.conv2d(prevLayerOut[prl], kernel, [1, 1, 1, 1], padding='SAME')

                if kwargs.get('weightNorm'):
                    # calc weight norm
                    conv = batch_norm('weight_norm', conv, dtype, kwargs.get('phase'))

                bias = tf.nn.bias_add(conv, biases)
                convReluPrl = tf.nn.relu(historicLayerOut[prl]+conv, name=scope.name)
                # Concatinate results along last dimension to get one output tensor
                if prl is 0:
                    convRelu = convReluPrl
                else:
                    convRelu = tf.concat([convRelu, convReluPrl], axis=3)

            _activation_summary(convRelu)

    return convRelu, numParallelModules*fireDimsSingleModule[cnnName]

def conv_fire_parallel_module(name, prevLayerOut, prevLayerDim, fireDimsSingleModule, numParallelModules, wd=None, **kwargs):
    """
    Input Args:
        name:               scope name
        prevLayerOut:       output tensor of previous layer
        prevLayerDim:       size of the last (3rd) dimension in prevLayerOut
        numParallelModules: number of parallel modules and parallel data in prevLayerOut
        fireDimsSingleModule:     number of output dimensions for each parallel module
    """
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')

    # Twin network -> numParallelModules = 2
    # Split tensor through last dimension into numParallelModules tensors
    prevLayerOut = tf.split(prevLayerOut, numParallelModules, axis=3)
    prevLayerIndivDims = prevLayerDim / numParallelModules

    if (fireDimsSingleModule.get('cnn1x1')):
        cnnName = 'cnn1x1'
        kernelSizeR = 1
        kernelSizeC = 1
    if (fireDimsSingleModule.get('cnn3x3')):
        cnnName = 'cnn3x3'
        kernelSizeR = 3
        kernelSizeC = 3
    if (fireDimsSingleModule.get('cnn5x5')):
        cnnName = 'cnn5x5'
        kernelSizeR = 5
        kernelSizeC = 5
    if (fireDimsSingleModule.get('cnn7x7')):
        cnnName = 'cnn7x7'
        kernelSizeR = 7
        kernelSizeC = 7

    with tf.variable_scope(name):
        with tf.variable_scope(cnnName) as scope:
            layerName = scope.name.replace("/", "_")
            #kernel = _variable_with_weight_decay('weights',
            #                                     shape=[kernelSizeR, kernelSizeR, prevLayerIndivDims, fireDimsSingleModule['cnn3x3']],
            #                                     initializer=existingParams[layerName]['weights'] if (existingParams is not None and
            #                                                                                         layerName in existingParams) else
            #                                                    (tf.contrib.layers.xavier_initializer_conv2d() if kwargs.get('phase')=='train' else
            #                                                     tf.constant_initializer(0.0, dtype=dtype)),
            #                                     dtype=dtype,
            # wd=wd,
            #                                     trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
            #                                                                               layerName in existingParams) else True)
            stddev = np.sqrt(2/np.prod(prevLayerOut[0].get_shape().as_list()[1:]))
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[kernelSizeR, kernelSizeR, prevLayerIndivDims, fireDimsSingleModule[cnnName]],
                                                 initializer=existingParams[layerName]['weights'] if (existingParams is not None and
                                                                                                     layerName in existingParams) else
                                                                (tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train' else
                                                                 tf.constant_initializer(0.0, dtype=dtype)),
                                                 dtype=dtype,
                                                 wd=wd,
                                                 trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            if existingParams is not None and layerName in existingParams:
                biases = tf.get_variable('biases',
                                         initializer=existingParams[layerName]['biases'], 
                                         dtype=dtype)
            else:
                biases = tf.get_variable('biases', [fireDimsSingleModule[cnnName]],
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         dtype=dtype)

            for prl in range(numParallelModules):
                conv = tf.nn.conv2d(prevLayerOut[prl], kernel, [1, 1, 1, 1], padding='SAME')

                if kwargs.get('weightNorm'):
                    # calc weight norm
                    conv = batch_norm('weight_norm', conv, dtype, kwargs.get('phase'))

                bias = tf.nn.bias_add(conv, biases)
                convReluPrl = tf.nn.relu(conv, name=scope.name)
                # Concatinate results along last dimension to get one output tensor
                if prl is 0:
                    convRelu = convReluPrl
                else:
                    convRelu = tf.concat([convRelu, convReluPrl], axis=3)

            _activation_summary(convRelu)

    return convRelu, numParallelModules*fireDimsSingleModule[cnnName]

def conv_fire_parallel_module_l2regul(name, prevLayerOut, prevLayerDim, 
                                      fireDimsSingleModule, numParallelModules,
                                      stride=[1, 1, 1, 1], 
                                      wd=None, **kwargs):
    """
    Input Args:
        name:               scope name
        prevLayerOut:       output tensor of previous layer
        prevLayerDim:       size of the last (3rd) dimension in prevLayerOut
        numParallelModules: number of parallel modules and parallel data in prevLayerOut
        fireDimsSingleModule:     number of output dimensions for each parallel module
    """
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')

    # Twin network -> numParallelModules = 2
    # Split tensor through last dimension into numParallelModules tensors
    prevLayerOut = tf.split(prevLayerOut, numParallelModules, axis=3)
    prevLayerIndivDims = prevLayerDim / numParallelModules

    if (fireDimsSingleModule.get('cnn1x1')):
        cnnName = 'cnn1x1'
        kernelSizeR = 1
        kernelSizeC = 1
    if (fireDimsSingleModule.get('cnn3x3')):
        cnnName = 'cnn3x3'
        kernelSizeR = 3
        kernelSizeC = 3
    if (fireDimsSingleModule.get('cnn5x5')):
        cnnName = 'cnn5x5'
        kernelSizeR = 5
        kernelSizeC = 5
    if (fireDimsSingleModule.get('cnn7x7')):
        cnnName = 'cnn7x7'
        kernelSizeR = 7
        kernelSizeC = 7

    print('----', name, cnnName, prevLayerDim, kernelSizeR, kernelSizeC, fireDimsSingleModule[cnnName], numParallelModules, prevLayerOut[0].get_shape())

    with tf.variable_scope(name):
        with tf.variable_scope(cnnName) as scope:
            layerName = scope.name.replace("/", "_")
            stddev = np.sqrt(2/np.prod(prevLayerOut[0].get_shape().as_list()[1:]))
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[kernelSizeR, kernelSizeR, prevLayerIndivDims, fireDimsSingleModule[cnnName]],
                                                 initializer=existingParams[layerName]['weights'] if (existingParams is not None and
                                                                                                     layerName in existingParams) else
                                                                (tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train' else
                                                                 tf.constant_initializer(0.0, dtype=dtype)),
                                                 dtype=dtype,
                                                 wd=wd,
                                                 trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            if existingParams is not None and layerName in existingParams:
                biases = tf.get_variable('biases',
                                         initializer=existingParams[layerName]['biases'], 
                                         dtype=dtype)
            else:
                biases = tf.get_variable('biases', [fireDimsSingleModule[cnnName]],
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         dtype=dtype)

            for prl in range(numParallelModules):
                conv = tf.nn.conv2d(prevLayerOut[prl], kernel, stride, padding='SAME')

                if kwargs.get('weightNorm'):
                    # calc weight norm
                    conv = batch_norm('weight_norm', conv, dtype, kwargs.get('phase'))

                bias = tf.nn.bias_add(conv, biases)
                convReluPrl = tf.nn.relu(conv, name=scope.name)
                # Concatinate results along last dimension to get one output tensor
                if prl is 0:
                    convRelu = convReluPrl
                else:
                    convRelu = tf.concat([convRelu, convReluPrl], axis=3)

            _activation_summary(convRelu)
            if kwargs.get('batchNorm'):
                convRelu = batch_norm('batchnorm', convRelu, dtype, kwargs.get('phase'))
            l2reg = tf.nn.l2_loss(kernel)

    return convRelu, numParallelModules*fireDimsSingleModule[cnnName], l2reg

def conv_fire_parallel_inception_module(name, prevLayerOut, prevLayerDim, fireDimsSingleModule, numParallelModules=2, wd=None, **kwargs):
    """
    Input Args:
        name:               scope name
        prevLayerOut:       output tensor of previous layer
        prevLayerDim:       size of the last (3rd) dimension in prevLayerOut
        numParallelModules: number of parallel modules and parallel data in prevLayerOut
        fireDimsSingleModule:     number of output dimensions for each parallel module
    """
    if (fireDimsSingleModule.get('cnn1x1')):
        fireOut_1x1, prevExpandDim_1x1 = conv_fire_parallel_module(name, prevLayerOut, prevLayerDim, {'cnn1x1': fireDimsSingleModule.get('cnn1x1')}, numParallelModules, wd, **kwargs)
    if (fireDimsSingleModule.get('cnn3x3')):
        fireOut_3x3, prevExpandDim_3x3 = conv_fire_parallel_module(name, prevLayerOut, prevLayerDim, {'cnn3x3': fireDimsSingleModule.get('cnn3x3')}, numParallelModules, wd, **kwargs)
    if (fireDimsSingleModule.get('cnn5x5')):
        fireOut_5x5, prevExpandDim_5x5 = conv_fire_parallel_module(name, prevLayerOut, prevLayerDim, {'cnn5x5': fireDimsSingleModule.get('cnn5x5')}, numParallelModules, wd, **kwargs)
    #if (fireDimsSingleModule.get('cnn7x7')):
    #    fireOut_1x1, prevExpandDim_1x1 = conv_fire_parallel_module(name, prevLayerOut, prevLayerDim, {'cnn7x7': fireDimsSingleModule.get('cnn7x7')}, numParallelModules, wd, **kwargs)
    
    fireOut = tf.concat([fireOut_1x1, fireOut_3x3, fireOut_5x5], axis=3)

    return fireOut, (prevExpandDim_1x1+prevExpandDim_3x3+prevExpandDim_5x5)

def conv_fire_parallel_inception_module_l2reg(name, prevLayerOut, prevLayerDim, fireDimsSingleModule, numParallelModules=2, wd=None, stride=[1,1,1,1], **kwargs):
    """
    Input Args:
        name:               scope name
        prevLayerOut:       output tensor of previous layer
        prevLayerDim:       size of the last (3rd) dimension in prevLayerOut
        numParallelModules: number of parallel modules and parallel data in prevLayerOut
        fireDimsSingleModule:     number of output dimensions for each parallel module
    """
    if (fireDimsSingleModule.get('cnn1x1')):
        fireOut_1x1, prevExpandDim_1x1, l2_1x1 = conv_fire_parallel_module_l2regul(name, prevLayerOut, prevLayerDim, {'cnn1x1': fireDimsSingleModule.get('cnn1x1')}, numParallelModules, wd, stride, **kwargs)
    if (fireDimsSingleModule.get('cnn3x3')):
        fireOut_3x3, prevExpandDim_3x3, l2_3x3 = conv_fire_parallel_module_l2regul(name, prevLayerOut, prevLayerDim, {'cnn3x3': fireDimsSingleModule.get('cnn3x3')}, numParallelModules, wd, stride, **kwargs)
    if (fireDimsSingleModule.get('cnn5x5')):
        fireOut_5x5, prevExpandDim_5x5, l2_5x5 = conv_fire_parallel_module_l2regul(name, prevLayerOut, prevLayerDim, {'cnn5x5': fireDimsSingleModule.get('cnn5x5')}, numParallelModules, wd, stride, **kwargs)
    #if (fireDimsSingleModule.get('cnn7x7')):
    #    fireOut_1x1, prevExpandDim_1x1, l2_7x7 = conv_fire_parallel_module_l2regul(name, prevLayerOut, prevLayerDim, {'cnn7x7': fireDimsSingleModule.get('cnn7x7')}, numParallelModules, wd, stride, **kwargs)
    
    fireOut = tf.concat([fireOut_1x1, fireOut_3x3, fireOut_5x5], axis=3)

    return fireOut, (prevExpandDim_1x1+prevExpandDim_3x3+prevExpandDim_5x5)

def conv_fire_residual_module(name, prevLayerOut, prevLayerDim, historicLayerOut, historicLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32
    
    existingParams = kwargs.get('existingParams')
    
    # output depth of the convolution should be same as historic
    if fireDims != historicLayerDim:
        # TO DO
        if kwargs.get('residualPadding') == "conv":
            # convlove historic data with a kernel of 1x1 and output size of prevLayerDim
            historicLayerOut = historicLayerOut
        if kwargs.get('residualPadding') == "zeroPad":
            # zero pad current historicLayerOut to the size of prevLayerDim
            historicLayerOut = historicLayerOut
    
    if (fireDims.get('cnn1x1')):
        cnnName = 'cnn1x1'
        kernelSizeR = 1
        kernelSizeC = 1
    if (fireDims.get('cnn3x3')):
        cnnName = 'cnn3x3'
        kernelSizeR = 3
        kernelSizeC = 3
    if (fireDims.get('cnn5x5')):
        cnnName = 'cnn5x5'
        kernelSizeR = 5
        kernelSizeC = 5
    if (fireDims.get('cnn7x7')):
        cnnName = 'cnn7x7'
        kernelSizeR = 7
        kernelSizeC = 7

    with tf.variable_scope(name):
        with tf.variable_scope(cnnName) as scope:
            layerName = scope.name.replace("/", "_")
            #kernel = _variable_with_weight_decay('weights',
            #                                     shape=[kernelSizeR, kernelSizeR, prevLayerDim, fireDims['cnn3x3']],
            #                                     initializer=existingParams[layerName]['weights'] if (existingParams is not None and
            #                                                                                         layerName in existingParams) else
            #                                                    (tf.contrib.layers.xavier_initializer_conv2d() if kwargs.get('phase')=='train' else
            #                                                     tf.constant_initializer(0.0, dtype=dtype)),
            #                                     dtype=dtype,
            # wd=wd,
            #                                     trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
            #                                                                               layerName in existingParams) else True)
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[kernelSizeR, kernelSizeR, prevLayerDim, fireDims[cnnName]],
                                                 initializer=existingParams[layerName]['weights'] if (existingParams is not None and
                                                                                                     layerName in existingParams) else
                                                                (tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train' else
                                                                 tf.constant_initializer(0.0, dtype=dtype)),
                                                 dtype=dtype,
                                                 wd=wd,
                                                 trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            conv = tf.nn.conv2d(prevLayerOut, kernel, [1, 1, 1, 1], padding='SAME')

            if kwargs.get('weightNorm'):
                # calc weight norm
                conv = batch_norm('weight_norm', conv, dtype, kwargs.get('phase'))

            # residual
            convRelu = tf.nn.relu(historicLayerOut+conv, name=scope.name)
            _activation_summary(convRelu)

        return convRelu, fireDims[cnnName]

def conv_fire_module(name, prevLayerOut, prevLayerDim, fireDims, wd=None, stride=[1, 1, 1, 1], padding='SAME', **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32
    
    existingParams = kwargs.get('existingParams')
    
    if (fireDims.get('cnnFC')):
        cnnName = 'cnnFC'
        kernelSizeR = int(prevLayerOut.get_shape()[1])
        kernelSizeC = int(prevLayerOut.get_shape()[2])
        padding = 'VALID'
    if (fireDims.get('cnn1x1')):
        cnnName = 'cnn1x1'
        kernelSizeR = 1
        kernelSizeC = 1
    if (fireDims.get('cnn3x3')):
        cnnName = 'cnn3x3'
        kernelSizeR = 3
        kernelSizeC = 3
    if (fireDims.get('cnn5x5')):
        cnnName = 'cnn5x5'
        kernelSizeR = 5
        kernelSizeC = 5
    if (fireDims.get('cnn7x7')):
        cnnName = 'cnn7x7'
        kernelSizeR = 7
        kernelSizeC = 7
    if (fireDims.get('cnn11x5')):
        cnnName = 'cnn11x5'
        kernelSizeR = 11
        kernelSizeC = 5
    if (fireDims.get('cnn3x5')):
        cnnName = 'cnn3x5'
        kernelSizeR = 3
        kernelSizeC = 5
    if (fireDims.get('cnn1x3')):
        cnnName = 'cnn1x3'
        kernelSizeR = 1
        kernelSizeC = 3
    if (fireDims.get('cnn3x1')):
        cnnName = 'cnn3x1'
        kernelSizeR = 3
        kernelSizeC = 1

    print('----', name, cnnName, prevLayerDim, kernelSizeR, kernelSizeC, fireDims[cnnName], prevLayerOut.get_shape())

    with tf.variable_scope(name):
        with tf.variable_scope(cnnName) as scope:
            layerName = scope.name.replace("/", "_")
            #kernel = _variable_with_weight_decay('weights',
            #                                     shape=[kernelSizeR, kernelSizeR, prevLayerDim, fireDims['cnn3x3']],
            #                                     initializer=existingParams[layerName]['weights'] if (existingParams is not None and
            #                                                                                         layerName in existingParams) else
            #                                                    (tf.contrib.layers.xavier_initializer_conv2d() if kwargs.get('phase')=='train' else
            #                                                     tf.constant_initializer(0.0, dtype=dtype)),
            #                                     dtype=dtype,
            # wd=wd,
            #                                     trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
            #                                                                               layerName in existingParams) else True)
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[kernelSizeR, kernelSizeC, prevLayerDim, fireDims[cnnName]],
                                                 initializer=existingParams[layerName]['weights'] if (existingParams is not None and
                                                                                                     layerName in existingParams) else
                                                                (tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train' else
                                                                 tf.constant_initializer(0.0, dtype=dtype)),
                                                 dtype=dtype,
                                                 wd=wd,
                                                 trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            conv = tf.nn.conv2d(prevLayerOut, kernel, stride, padding=padding)

            if kwargs.get('weightNorm'):
                # calc weight norm
                conv = batch_norm('weight_norm', conv, dtype, kwargs.get('phase'))

            with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
                if existingParams is not None and layerName in existingParams:
                    biases = tf.get_variable('biases',
                                             initializer=existingParams[layerName]['biases'], dtype=dtype)
                else:
                    biases = tf.get_variable('biases', [fireDims[cnnName]],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=dtype)

            conv = tf.nn.bias_add(conv, biases)
            convRelu = tf.nn.relu(conv, name=scope.name)
            _activation_summary(convRelu)

            if kwargs.get('batchNorm'):
                convRelu = batch_norm('batchnorm', convRelu, dtype, kwargs.get('phase'))

        return convRelu, fireDims[cnnName]

def conv_fire_module_l2regul(name, prevLayerOut, prevLayerDim, 
                             fireDims, wd=None, 
                             stride=[1, 1, 1, 1], 
                             padding='SAME', **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32
    
    existingParams = kwargs.get('existingParams')
    
    if (fireDims.get('cnnFC')):
        cnnName = 'cnnFC'
        kernelSizeR = int(prevLayerOut.get_shape()[1])
        kernelSizeC = int(prevLayerOut.get_shape()[2])
        padding = 'VALID'
    if (fireDims.get('cnn1x1')):
        cnnName = 'cnn1x1'
        kernelSizeR = 1
        kernelSizeC = 1
    if (fireDims.get('cnn3x3')):
        cnnName = 'cnn3x3'
        kernelSizeR = 3
        kernelSizeC = 3
    if (fireDims.get('cnn5x5')):
        cnnName = 'cnn5x5'
        kernelSizeR = 5
        kernelSizeC = 5
    if (fireDims.get('cnn7x7')):
        cnnName = 'cnn7x7'
        kernelSizeR = 7
        kernelSizeC = 7
    if (fireDims.get('cnn11x5')):
        cnnName = 'cnn11x5'
        kernelSizeR = 11
        kernelSizeC = 5
    if (fireDims.get('cnn3x5')):
        cnnName = 'cnn3x5'
        kernelSizeR = 3
        kernelSizeC = 5
    if (fireDims.get('cnn1x3')):
        cnnName = 'cnn1x3'
        kernelSizeR = 1
        kernelSizeC = 3
    if (fireDims.get('cnn3x1')):
        cnnName = 'cnn3x1'
        kernelSizeR = 3
        kernelSizeC = 1

    print('----', name, cnnName, prevLayerDim, kernelSizeR, kernelSizeC, fireDims[cnnName], prevLayerOut.get_shape())

    with tf.variable_scope(name):
        with tf.variable_scope(cnnName) as scope:
            layerName = scope.name.replace("/", "_")
            #kernel = _variable_with_weight_decay('weights',
            #                                     shape=[kernelSizeR, kernelSizeR, prevLayerDim, fireDims['cnn3x3']],
            #                                     initializer=existingParams[layerName]['weights'] if (existingParams is not None and
            #                                                                                         layerName in existingParams) else
            #                                                    (tf.contrib.layers.xavier_initializer_conv2d() if kwargs.get('phase')=='train' else
            #                                                     tf.constant_initializer(0.0, dtype=dtype)),
            #                                     dtype=dtype,
            # wd=wd,
            #                                     trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
            #                                                                               layerName in existingParams) else True)
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[kernelSizeR, kernelSizeC, prevLayerDim, fireDims[cnnName]],
                                                 initializer=existingParams[layerName]['weights'] if (existingParams is not None and
                                                                                                     layerName in existingParams) else
                                                                (tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train' else
                                                                 tf.constant_initializer(0.0, dtype=dtype)),
                                                 dtype=dtype,
                                                 wd=wd,
                                                 trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            conv = tf.nn.conv2d(prevLayerOut, kernel, stride, padding=padding)

            if kwargs.get('weightNorm'):
                # calc weight norm
                conv = batch_norm('weight_norm', conv, dtype, kwargs.get('phase'))

            with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
                if existingParams is not None and layerName in existingParams:
                    biases = tf.get_variable('biases',
                                             initializer=existingParams[layerName]['biases'], dtype=dtype)
                else:
                    biases = tf.get_variable('biases', [fireDims[cnnName]],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=dtype)

            conv = tf.nn.bias_add(conv, biases)
            convRelu = tf.nn.relu(conv, name=scope.name)
            _activation_summary(convRelu)

            if kwargs.get('batchNorm'):
                convRelu = batch_norm('batchnorm', convRelu, dtype, kwargs.get('phase'))
            l2reg = tf.nn.l2_loss(kernel)
        return convRelu, fireDims[cnnName], l2reg

def conv_fire_module_sep(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32
    
    existingParams = kwargs.get('existingParams')
    
    padding = 'SAME'
    if (fireDims.get('cnnFC')):
        cnnName = 'cnnFC'
        kernelSizeR = int(prevLayerOut.get_shape()[1])
        kernelSizeC = int(prevLayerOut.get_shape()[2])
        padding = 'VALID'
    if (fireDims.get('cnn1x1')):
        cnnName = 'cnn1x1'
        kernelSizeR = 1
        kernelSizeC = 1
    if (fireDims.get('cnn3x3')):
        cnnName = 'cnn3x3'
        kernelSizeR = 3
        kernelSizeC = 3
    if (fireDims.get('cnn5x5')):
        cnnName = 'cnn5x5'
        kernelSizeR = 5
        kernelSizeC = 5
    if (fireDims.get('cnn7x7')):
        cnnName = 'cnn7x7'
        kernelSizeR = 7
        kernelSizeC = 7
    if (fireDims.get('cnn11x5')):
        cnnName = 'cnn11x5'
        kernelSizeR = 11
        kernelSizeC = 5
    if (fireDims.get('cnn3x5')):
        cnnName = 'cnn3x5'
        kernelSizeR = 3
        kernelSizeC = 5
    if (fireDims.get('cnn1x3')):
        cnnName = 'cnn1x3'
        kernelSizeR = 1
        kernelSizeC = 3

    print('----', name, cnnName, prevLayerDim, kernelSizeR, kernelSizeC, fireDims[cnnName])

    with tf.variable_scope(name):
        with tf.variable_scope(cnnName) as scope:
            layerName = scope.name.replace("/", "_")
            #kernel = _variable_with_weight_decay('weights',
            #                                     shape=[kernelSizeR, kernelSizeR, prevLayerDim, fireDims['cnn3x3']],
            #                                     initializer=existingParams[layerName]['weights'] if (existingParams is not None and
            #                                                                                         layerName in existingParams) else
            #                                                    (tf.contrib.layers.xavier_initializer_conv2d() if kwargs.get('phase')=='train' else
            #                                                     tf.constant_initializer(0.0, dtype=dtype)),
            #                                     dtype=dtype,
            # wd=wd,
            #                                     trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
            #                                                                               layerName in existingParams) else True)
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[kernelSizeR, kernelSizeC, prevLayerDim, fireDims[cnnName]],
                                                 initializer=existingParams[layerName]['weights'] if (existingParams is not None and
                                                                                                     layerName in existingParams) else
                                                                (tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train' else
                                                                 tf.constant_initializer(0.0, dtype=dtype)),
                                                 dtype=dtype,
                                                 wd=wd,
                                                 trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            #conv = tf.nn.conv2d(prevLayerOut, kernel, [1, 1, 1, 1], padding=padding)
            #conv = tf.nn.separable_conv2d(prevLayerOut, tf.constant([kernelSizeR, kernelSizeC, prevLayerDim, 1]), tf.constant([1,1, 1*prevLayerDim,fireDims[cnnName]]), [1,1,1,1], padding=padding)
            conv = tf.layers.separable_conv2d(prevLayerOut, fireDims[cnnName], [kernelSizeR, kernelSizeC], padding=padding)
            if kwargs.get('weightNorm'):
                # calc weight norm
                conv = batch_norm('weight_norm', conv, dtype, kwargs.get('phase'))

            with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
                if existingParams is not None and layerName in existingParams:
                    biases = tf.get_variable('biases',
                                             initializer=existingParams[layerName]['biases'], dtype=dtype)
                else:
                    biases = tf.get_variable('biases', [fireDims[cnnName]],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=dtype)

            conv = tf.nn.bias_add(conv, biases)
            convRelu = tf.nn.relu(conv, name=scope.name)
            _activation_summary(convRelu)

        return convRelu, fireDims[cnnName]

def conv_fire_inception_module(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    if (fireDims.get('cnnFC')):
        fireOut, prevExpandDim = conv_fire_module(name+'_inc_1x1_0', prevLayerOut, prevLayerDim, {'cnn1x1': prevLayerDim}, wd, stride=[1,1,1,1], **kwargs)
        fireOut, prevExpandDim = conv_fire_module(name+'_inc_3x3_1', fireOut, prevExpandDim, {'cnn3x3': prevExpandDim}, wd, stride=[1,2,2,1], padding = 'VALID', **kwargs)
        fireOut, prevExpandDim = conv_fire_module(name+'_inc_3x3_2', fireOut, prevExpandDim, {'cnn3x3': prevExpandDim}, wd, stride=[1,2,2,1], padding = 'VALID', **kwargs)
        fireOut, prevExpandDim = conv_fire_module(name+'_inc_1x3_3', fireOut, prevExpandDim, {'cnn1x3': prevExpandDim}, wd, stride=[1,1,2,1], **kwargs)
        fireOut, prevExpandDim = conv_fire_module(name+'_inc_3x1_4', fireOut, prevExpandDim, {'cnn3x1': fireDims.get('cnnFC')}, wd, stride=[1,2,1,1], **kwargs)
        #fireOut, prevExpandDim = conv_fire_module(name+'_inc_1x3_5', fireOut, prevExpandDim, {'cnn1x3': prevExpandDim}, wd, stride=[1,1,2,1], **kwargs)
        #fireOut, prevExpandDim = conv_fire_module(name+'_inc_3x1_6', fireOut, prevExpandDim, {'cnn3x1': fireDims.get('cnnFC')}, wd, stride=[1,2,1,1], **kwargs)

    if (fireDims.get('cnn1x1')):
        fireOut_1x1, prevExpandDim_1x1 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn1x1': fireDims.get('cnn1x1')}, wd, **kwargs)
    if (fireDims.get('cnn3x3')):
        fireOut_3x3, prevExpandDim_3x3 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn3x3': fireDims.get('cnn3x3')}, wd, **kwargs)
    if (fireDims.get('cnn5x5')):
        fireOut_5x5, prevExpandDim_5x5 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn5x5': fireDims.get('cnn5x5')}, wd, **kwargs)
    #if (fireDims.get('cnn7x7')):
    #    fireOut_1x1, prevExpandDim_1x1 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn7x7': fireDims.get('cnn7x7')}, wd, **kwargs)
    
    if (fireDims.get('cnn1x1') and fireDims.get('cnn3x3') and fireDims.get('cnn5x5')):
        fireOut = tf.concat([fireOut_1x1, fireOut_3x3, fireOut_5x5], axis=3)
        prevExpandDim = prevExpandDim_1x1+prevExpandDim_3x3+prevExpandDim_5x5
    elif (fireDims.get('cnn3x3') and fireDims.get('cnn5x5')):
        fireOut = tf.concat([fireOut_3x3, fireOut_5x5], axis=3)
        prevExpandDim = prevExpandDim_3x3+prevExpandDim_5x5
    elif (fireDims.get('cnn1x1') and fireDims.get('cnn3x3')):
        fireOut = tf.concat([fireOut_1x1, fireOut_3x3], axis=3)
        prevExpandDim = prevExpandDim_1x1+prevExpandDim_3x3
    elif (fireDims.get('cnn1x1') and fireDims.get('cnn5x5')):
        fireOut = tf.concat([fireOut_1x1, fireOut_5x5], axis=3)
        prevExpandDim = prevExpandDim_1x1+prevExpandDim_5x5
    elif fireDims.get('cnn1x1'):
        fireOut = fireOut_1x1
        prevExpandDim = prevExpandDim_1x1
    elif fireDims.get('cnn3x3'):
        fireOut = fireOut_3x3
        prevExpandDim = prevExpandDim_3x3
    elif fireDims.get('cnn5x5'):
        fireOut = fireOut_5x5
        prevExpandDim = prevExpandDim_5x5
    
    return fireOut, prevExpandDim

def conv_fire_inception_module_l2reg(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    if (fireDims.get('cnnFC')):
        # X x Y
        fireOut, prevExpandDim, l2regTemp = conv_fire_module_l2regul(name+'_inc_1x1_0', prevLayerOut, prevLayerDim, {'cnn1x1': fireDims['cnnFC']}, wd, stride=[1,1,1,1], **kwargs)
        l2reg = l2regTemp
        # X x Y
        fireOut, prevExpandDim, l2regTemp = conv_fire_module_l2regul(name+'_inc_3x3_2', fireOut, prevExpandDim, {'cnn3x3': fireDims['cnnFC']}, wd, stride=[1,2,2,1], padding = 'SAME', **kwargs)
        l2reg += l2regTemp
        # X/2 x Y/2
        fireOut, prevExpandDim, l2regTemp = conv_fire_module_l2regul(name+'_inc_1x3_1', fireOut, prevExpandDim, {'cnn1x3': fireDims['cnnFC']}, wd, stride=[1,1,2,1], padding = 'SAME', **kwargs)
        l2reg += l2regTemp
        # X/4 x Y/2
        l2reg = l2reg/3

    if (fireDims.get('cnn1x1')):
        fireOut_1x1, prevExpandDim_1x1 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn1x1': fireDims.get('cnn1x1')}, wd, **kwargs)
    if (fireDims.get('cnn3x3')):
        fireOut_3x3, prevExpandDim_3x3 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn3x3': fireDims.get('cnn3x3')}, wd, **kwargs)
    if (fireDims.get('cnn5x5')):
        fireOut_5x5, prevExpandDim_5x5 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn5x5': fireDims.get('cnn5x5')}, wd, **kwargs)
    #if (fireDims.get('cnn7x7')):
    #    fireOut_1x1, prevExpandDim_1x1 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn7x7': fireDims.get('cnn7x7')}, wd, **kwargs)
    
    if (fireDims.get('cnn1x1') and fireDims.get('cnn3x3') and fireDims.get('cnn5x5')):
        fireOut = tf.concat([fireOut_1x1, fireOut_3x3, fireOut_5x5], axis=3)
        prevExpandDim = prevExpandDim_1x1+prevExpandDim_3x3+prevExpandDim_5x5
    elif (fireDims.get('cnn3x3') and fireDims.get('cnn5x5')):
        fireOut = tf.concat([fireOut_3x3, fireOut_5x5], axis=3)
        prevExpandDim = prevExpandDim_3x3+prevExpandDim_5x5
    elif (fireDims.get('cnn1x1') and fireDims.get('cnn3x3')):
        fireOut = tf.concat([fireOut_1x1, fireOut_3x3], axis=3)
        prevExpandDim = prevExpandDim_1x1+prevExpandDim_3x3
    elif (fireDims.get('cnn1x1') and fireDims.get('cnn5x5')):
        fireOut = tf.concat([fireOut_1x1, fireOut_5x5], axis=3)
        prevExpandDim = prevExpandDim_1x1+prevExpandDim_5x5
    elif fireDims.get('cnn1x1'):
        fireOut = fireOut_1x1
        prevExpandDim = prevExpandDim_1x1
    elif fireDims.get('cnn3x3'):
        fireOut = fireOut_3x3
        prevExpandDim = prevExpandDim_3x3
    elif fireDims.get('cnn5x5'):
        fireOut = fireOut_5x5
        prevExpandDim = prevExpandDim_5x5
    
    return fireOut, prevExpandDim, l2reg

def conv_par_inception_lin_module_l2reg(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    if (fireDims.get('cnnFC')):
        # X x Y
        fireOut, prevExpandDim, l2regTemp = conv_fire_module_l2regul(name+'_inc_1x1_0', prevLayerOut, prevLayerDim, {'cnn1x1': fireDims['cnnFC']}, wd, stride=[1,1,1,1], **kwargs)
        l2reg = l2regTemp
        # X x Y
        fireOut, prevExpandDim, l2regTemp = conv_fire_module_l2regul(name+'_inc_3x3_2', fireOut, prevExpandDim, {'cnn3x3': fireDims['cnnFC']}, wd, stride=[1,2,2,1], padding = 'SAME', **kwargs)
        l2reg += l2regTemp
        # X/2 x Y/2
        fireOut, prevExpandDim, l2regTemp = conv_fire_module_l2regul(name+'_inc_1x3_1', fireOut, prevExpandDim, {'cnn1x3': fireDims['cnnFC']}, wd, stride=[1,1,2,1], padding = 'SAME', **kwargs)
        l2reg += l2regTemp
        # X/4 x Y/2
        l2reg = l2reg/3

    if (fireDims.get('cnn1x1')):
        fireOut_1x1, prevExpandDim_1x1 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn1x1': fireDims.get('cnn1x1')}, wd, **kwargs)
    if (fireDims.get('cnn3x3')):
        fireOut_3x3, prevExpandDim_3x3 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn3x3': fireDims.get('cnn3x3')}, wd, **kwargs)
    if (fireDims.get('cnn5x5')):
        fireOut_5x5, prevExpandDim_5x5 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn5x5': fireDims.get('cnn5x5')}, wd, **kwargs)
    #if (fireDims.get('cnn7x7')):
    #    fireOut_1x1, prevExpandDim_1x1 = conv_fire_module(name, prevLayerOut, prevLayerDim, {'cnn7x7': fireDims.get('cnn7x7')}, wd, **kwargs)
    
    if (fireDims.get('cnn1x1') and fireDims.get('cnn3x3') and fireDims.get('cnn5x5')):
        fireOut = tf.concat([fireOut_1x1, fireOut_3x3, fireOut_5x5], axis=3)
        prevExpandDim = prevExpandDim_1x1+prevExpandDim_3x3+prevExpandDim_5x5
    elif (fireDims.get('cnn3x3') and fireDims.get('cnn5x5')):
        fireOut = tf.concat([fireOut_3x3, fireOut_5x5], axis=3)
        prevExpandDim = prevExpandDim_3x3+prevExpandDim_5x5
    elif (fireDims.get('cnn1x1') and fireDims.get('cnn3x3')):
        fireOut = tf.concat([fireOut_1x1, fireOut_3x3], axis=3)
        prevExpandDim = prevExpandDim_1x1+prevExpandDim_3x3
    elif (fireDims.get('cnn1x1') and fireDims.get('cnn5x5')):
        fireOut = tf.concat([fireOut_1x1, fireOut_5x5], axis=3)
        prevExpandDim = prevExpandDim_1x1+prevExpandDim_5x5
    elif fireDims.get('cnn1x1'):
        fireOut = fireOut_1x1
        prevExpandDim = prevExpandDim_1x1
    elif fireDims.get('cnn3x3'):
        fireOut = fireOut_3x3
        prevExpandDim = prevExpandDim_3x3
    elif fireDims.get('cnn5x5'):
        fireOut = fireOut_5x5
        prevExpandDim = prevExpandDim_5x5
    
    return fireOut, prevExpandDim, l2reg

def fc_fire_module(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')

    with tf.variable_scope(name):
        with tf.variable_scope('fc') as scope:
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            fcWeights = _variable_with_weight_decay('weights',
                                                    shape=[prevLayerDim, fireDims['fc']],
                                                    initializer=(tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train'
                                                                   else tf.constant_initializer(0.0, dtype=dtype)),
                                                    dtype=dtype,
                                                    wd=wd,
                                                    trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            # prevLayerOut is [batchSize, HxWxD], matmul -> [batchSize, fireDims['fc']]
            fc = tf.matmul(prevLayerOut, fcWeights)

            if kwargs.get('weightNorm'):
                # calc weight norm
                fc = batch_norm('weight_norm', fc, dtype, kwargs.get('phase'))
            
            with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
                biases = tf.get_variable('biases', fireDims['fc'],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)
            fc = tf.nn.bias_add(fc, biases)
            fcRelu = tf.nn.relu(fc, name=scope.name)
            
            if kwargs.get('batchNorm'):
                fcRelu = batch_norm('batchnorm', fcRelu, dtype, kwargs.get('phase'))
            _activation_summary(fcRelu)

        return fcRelu, fireDims['fc']

def fc_fire_module_l2regul(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')
    
    print('----', name, 'FC', prevLayerDim, fireDims['fc'])


    with tf.variable_scope(name):
        with tf.variable_scope('fc') as scope:
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            fcWeights = _variable_with_weight_decay('weights',
                                                    shape=[prevLayerDim, fireDims['fc']],
                                                    initializer=(tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train'
                                                                   else tf.constant_initializer(0.0, dtype=dtype)),
                                                    dtype=dtype,
                                                    wd=wd,
                                                    trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            # prevLayerOut is [batchSize, HxWxD], matmul -> [batchSize, fireDims['fc']]
            fc = tf.matmul(prevLayerOut, fcWeights)

            if kwargs.get('weightNorm'):
                # calc weight norm
                fc = batch_norm('weight_norm', fc, dtype, kwargs.get('phase'))
            
            with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
                biases = tf.get_variable('biases', fireDims['fc'],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)
            fc = tf.nn.bias_add(fc, biases)
            fcRelu = tf.nn.relu(fc, name=scope.name)
            
            if kwargs.get('batchNorm'):
                fcRelu = batch_norm('batchnorm', fcRelu, dtype, kwargs.get('phase'))
            _activation_summary(fcRelu)
            l2reg = tf.nn.l2_loss(fcWeights)

        return fcRelu, fireDims['fc'], l2reg

def fc_fire_parallel_module_l2regul(name, prevLayerOut, prevLayerDim, fireDims, numParallelModules, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')
    
    # Twin network -> numParallelModules = 2
    # Split tensor through last dimension into numParallelModules tensors
    prevLayerOut = tf.split(prevLayerOut, numParallelModules, axis=3)
    prevLayerIndivDims = prevLayerDim / numParallelModules

    print('----', name, 'FC', prevLayerDim, fireDims['fc'], numParallelModules)


    with tf.variable_scope(name):
        with tf.variable_scope('fc') as scope:
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            fcWeights = _variable_with_weight_decay('weights',
                                                    shape=[prevLayerDim, fireDims['fc']],
                                                    initializer=(tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train'
                                                                   else tf.constant_initializer(0.0, dtype=dtype)),
                                                    dtype=dtype,
                                                    wd=wd,
                                                    trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            for prl in range(numParallelModules):
                # prevLayerOut is [batchSize, HxWxD], matmul -> [batchSize, fireDims['fc']]
                fc = tf.matmul(prevLayerOut[prl], fcWeights)
                
                if kwargs.get('weightNorm'):
                    # calc weight norm
                    fc = batch_norm('weight_norm', fc, dtype, kwargs.get('phase'))

                with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
                biases = tf.get_variable('biases', fireDims['fc'],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)

                fcReluPrl = tf.nn.relu(fc, name=scope.name)
                # Concatinate results along last dimension to get one output tensor
                if prl is 0:
                    fcRelu = fcReluPrl
                else:
                    fcRelu = tf.concat([fcRelu, fcReluPrl], axis=3)

            _activation_summary(convRelu)
            if kwargs.get('batchNorm'):
                fcRelu = batch_norm('batchnorm', fcRelu, dtype, kwargs.get('phase'))

            l2reg = tf.nn.l2_loss(fcWeights)
            _activation_summary(fcRelu)

        return fcRelu, fireDims['fc'], l2reg

def fc_fire_LSTM_module(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')

    with tf.variable_scope(name):
        with tf.variable_scope('fc') as scope:
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            fcWeights = _variable_with_weight_decay('weights',
                                                    shape=[prevLayerDim, fireDims['fc']],
                                                    initializer=(tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train'
                                                                   else tf.constant_initializer(0.0, dtype=dtype)),
                                                    dtype=dtype,
                                                    wd=wd,
                                                    trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            # prevLayerOut is [batchSize, HxWxD], matmul -> [batchSize, fireDims['fc']]
            fc = tf.matmul(prevLayerOut, fcWeights)

            if kwargs.get('weightNorm'):
                # calc weight norm
                fc = batch_norm('weight_norm', fc, dtype, kwargs.get('phase'))

            biases = tf.get_variable('biases', fireDims['fc'],
                                     initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)
            fc = tf.nn.bias_add(fc, biases)
            fcRelu = tf.nn.relu(fc, name=scope.name)
            _activation_summary(fcRelu)
        
        return fcRelu, fireDims['fc']
def fc_regression_module(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')

    with tf.variable_scope(name):
        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            fcWeights = _variable_with_weight_decay('weights',
                                                    shape=[prevLayerDim, fireDims['fc']],
                                                    initializer=(tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train'
                                                                   else tf.constant_initializer(0.0, dtype=dtype)),
                                                    dtype=dtype,
                                                    wd=wd,
                                                    trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            # prevLayerOut is [batchSize, HxWxD], matmul -> [batchSize, fireDims['fc']]
            fc = tf.matmul(prevLayerOut, fcWeights)

            if kwargs.get('weightNorm'):
                # calc weight norm
                fc = batch_norm('weight_norm', fc, dtype, kwargs.get('phase'))

            biases = tf.get_variable('biases', fireDims['fc'],
                                     initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)
            fc = tf.nn.bias_add(fc, biases)
            _activation_summary(fc)
        
        return fc, fireDims['fc']

def fc_regression_module_l2regul(name, prevLayerOut, prevLayerDim, fireDims, wd=None, **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    existingParams = kwargs.get('existingParams')

    with tf.variable_scope(name):
        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
            stddev = np.sqrt(2/np.prod(prevLayerOut.get_shape().as_list()[1:]))
            fcWeights = _variable_with_weight_decay('weights',
                                                    shape=[prevLayerDim, fireDims['fc']],
                                                    initializer=(tf.contrib.layers.xavier_initializer() if kwargs.get('phase')=='train'
                                                                   else tf.constant_initializer(0.0, dtype=dtype)),
                                                    dtype=dtype,
                                                    wd=wd,
                                                    trainable=kwargs.get('tuneExistingWeights') if (existingParams is not None and 
                                                                                           layerName in existingParams) else True)
            
            # prevLayerOut is [batchSize, HxWxD], matmul -> [batchSize, fireDims['fc']]
            fc = tf.matmul(prevLayerOut, fcWeights)

            if kwargs.get('weightNorm'):
                # calc weight norm
                fc = batch_norm('weight_norm', fc, dtype, kwargs.get('phase'))

            biases = tf.get_variable('biases', fireDims['fc'],
                                     initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)
            fc = tf.nn.bias_add(fc, biases)
            _activation_summary(fc)
            l2reg = tf.nn.l2_loss(fcWeights)
        return fc, fireDims['fc'], l2reg


def deconv_fire_module(name, prevLayerOut, prevLayerDim, fireDims, wd=None, stride=[1, 1, 1, 1], padding='SAME', **kwargs):
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32
    
    existingParams = kwargs.get('existingParams')
    
    if (fireDims.get('deConv1x1')):
        cnnName = 'deConv1x1'
        kernelSizeR = 1
        kernelSizeC = 1
    if (fireDims.get('deConv3x3')):
        cnnName = 'deConv3x3'
        kernelSizeR = 3
        kernelSizeC = 3
    if (fireDims.get('deConv5x5')):
        cnnName = 'deConv5x5'
        kernelSizeR = 5
        kernelSizeC = 5
    if (fireDims.get('deConv7x7')):
        cnnName = 'deConv7x7'
        kernelSizeR = 7
        kernelSizeC = 7
    if (fireDims.get('deConv11x5')):
        cnnName = 'deConv11x5'
        kernelSizeR = 11
        kernelSizeC = 5
    if (fireDims.get('deConv3x5')):
        cnnName = 'deConv3x5'
        kernelSizeR = 3
        kernelSizeC = 5
    if (fireDims.get('deConv1x3')):
        cnnName = 'deConv1x3'
        kernelSizeR = 1
        kernelSizeC = 3
    if (fireDims.get('deConv3x1')):
        cnnName = 'deConv3x1'
        kernelSizeR = 3
        kernelSizeC = 1

    print('----', name, cnnName, prevLayerDim, kernelSizeR, kernelSizeC, fireDims[cnnName], prevLayerOut.get_shape())

    with tf.variable_scope(name):
        with tf.variable_scope(cnnName) as scope:
            layerName = scope.name.replace("/", "_")
            stride=(1, 1)
            padding='valid'
            deConv = tf.nn.conv2d_transpose(inputs=prevLayerOut, filters=fireDims[cnnName], kernel_size=[kernelSizeR, kernelSizeC],
                                          strides=stride, padding=padding, activation='relu', name=layerName)
            
            if kwargs.get('batchNorm'):
                deConv = batch_norm('batchnorm', deConv, dtype, kwargs.get('phase'))

        return deConv, fireDims[cnnName]
        


def loss(pred, tval, **kwargs):
    return loss_base.loss(pred, tval, **kwargs)

def loss_l2reg(pred, tval, l2reg, **kwargs):
    return loss_base.loss_l2reg(pred, tval, l2reg, **kwargs)

def train(loss, globalStep, **kwargs):
    if kwargs.get('optimizer') == 'MomentumOptimizer':
        optimizerParams = optimizer_params.get_momentum_optimizer_params(globalStep, **kwargs)
    if kwargs.get('optimizer') == 'AdamOptimizer':
        optimizerParams = optimizer_params.get_adam_optimizer_params(globalStep, **kwargs)
    if kwargs.get('optimizer') == 'GradientDescentOptimizer':
        optimizerParams = optimizer_params.get_gradient_descent_optimizer_params(globalStep, **kwargs)
    if kwargs.get('optimizer') == 'AdaGrad':
        optimizerParams = optimizer_params.get_adagrad_optimizer_params(globalStep, **kwargs)

    # Generate moving averages of all losses and associated summaries.
    loss_base.add_loss_summaries(loss, kwargs.get('activeBatchSize', None))
    
    # Compute gradients.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batchnorm
    tvars = tf.trainable_variables()
    with tf.control_dependencies(update_ops):
        if kwargs.get('optimizer') == 'AdamOptimizer':
            optim = tf.train.AdamOptimizer(learning_rate=optimizerParams['learningRate'], epsilon=optimizerParams['epsilon'])
        if kwargs.get('optimizer') == 'MomentumOptimizer':
            optim = tf.train.MomentumOptimizer(learning_rate=optimizerParams['learningRate'], momentum=optimizerParams['momentum'])
        if kwargs.get('optimizer') == 'GradientDescentOptimizer':
            optim = tf.train.GradientDescentOptimizer(learning_rate=optimizerParams['learningRate'])
        if kwargs.get('optimizer') == 'AdaGrad':
            optim = tf.train.AdamOptimizer(learning_rate=optimizerParams['learningRate'])

        grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), kwargs.get('clipNorm'))

    # Apply gradients.
    #applyGradientOp = opt.apply_gradients(grads, global_step=globalStep)
    #train_op = opt.apply_gradients(gradsNvars, global_step=globalStep)
    opApplyGradients = optim.apply_gradients(zip(grads, tvars), global_step=globalStep)

        
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():    
        tf.summary.histogram(var.op.name, var)
    
    # Add histograms for gradients.
    for grad, var in zip(grads, tvars):
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    
    with tf.control_dependencies([opApplyGradients]):
        opTrain = tf.no_op(name='train')
    return opTrain

def test(loss, globalStep, **kwargs):
    # Generate moving averages of all losses and associated summaries.
    lossAveragesOp = loss_base.add_loss_summaries(loss, kwargs.get('activeBatchSize', None))
    
    with tf.control_dependencies([]):
        opTest = tf.no_op(name='test')

    return opTest
