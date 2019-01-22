from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

def _get_learning_rate_piecewise_constant(globalStep, **kwargs):
    # piecewise constant learning rate
    # 0.005 for [0,30000] -> 0.0005 for [30001,60000], 0.00005 for [60001, 90000]
    # [30000, 60000]
    boundaries = [kwargs.get('numEpochsPerDecay'),
                  2*kwargs.get('numEpochsPerDecay')]
    #[0.005, 0.0005, 0.00005]
    values = [kwargs.get('initialLearningRate'), 
              kwargs.get('initialLearningRate')*kwargs.get('learningRateDecayFactor'),
              kwargs.get('initialLearningRate')*kwargs.get('learningRateDecayFactor')*kwargs.get('learningRateDecayFactor')]
    # piecewise constant learning rate
    learningRate = tf.train.piecewise_constant(globalStep, boundaries, values)
    tf.summary.scalar('learning_rate', learningRate)
    return learningRate

def _get_learning_rate_piecewise_shifted(globalStep, **kwargs):
    # piecewise shifted learning rate
    #boundaries = [float(int(kwargs.get('trainMaxSteps')*0.15)),
    #              float(int(kwargs.get('trainMaxSteps')*0.55)),
    #              float(int(kwargs.get('trainMaxSteps')*0.80)),
    #              float(int(kwargs.get('trainMaxSteps')*0.90)),
    #              float(int(kwargs.get('trainMaxSteps')*0.95)),
    #              float(int(kwargs.get('trainMaxSteps')*0.97)),
    #              float(int(kwargs.get('trainMaxSteps')*0.98)),
    #              float(int(kwargs.get('trainMaxSteps')*0.99))]
    boundaries = [float(int(kwargs.get('trainMaxSteps')*0.30)),
                  float(int(kwargs.get('trainMaxSteps')*0.50)),
                  float(int(kwargs.get('trainMaxSteps')*0.65)),
                  float(int(kwargs.get('trainMaxSteps')*0.70)),
                  float(int(kwargs.get('trainMaxSteps')*0.80)),
                  float(int(kwargs.get('trainMaxSteps')*0.87)),
                  float(int(kwargs.get('trainMaxSteps')*0.92)),
                  float(int(kwargs.get('trainMaxSteps')*0.96))]
    decay = kwargs.get('learningRateDecayFactor')
    values = [kwargs.get('initialLearningRate'),
              kwargs.get('initialLearningRate')*decay,
              kwargs.get('initialLearningRate')*decay*decay,
              kwargs.get('initialLearningRate')*decay*decay*decay,
              kwargs.get('initialLearningRate')*decay*decay*decay*decay,
              kwargs.get('initialLearningRate')*decay*decay*decay*decay*decay,
              kwargs.get('initialLearningRate')*decay*decay*decay*decay*decay*decay,
              kwargs.get('initialLearningRate')*decay*decay*decay*decay*decay*decay*decay,
              kwargs.get('initialLearningRate')*decay*decay*decay*decay*decay*decay*decay*decay]
    # piecewise constant learning rate
    learningRate = tf.train.piecewise_constant(globalStep, boundaries, values)
    tf.summary.scalar('learning_rate', learningRate)
    return learningRate

def get_momentum_optimizer_params(globalStep, **kwargs):
    learningRate = _get_learning_rate_piecewise_constant(globalStep, **kwargs)
    momentum = kwargs.get('momentum')
    return {'learningRate': learningRate, 'momentum': momentum}

def get_adam_optimizer_params(globalStep, **kwargs):
    learningRate = _get_learning_rate_piecewise_constant(globalStep, **kwargs)
    epsilon = kwargs.get('epsilon')
    return {'learningRate': learningRate, 'epsilon': epsilon}

def get_gradient_descent_optimizer_params(globalStep, **kwargs):
    learningRate = _get_learning_rate_piecewise_constant(globalStep, **kwargs)
    return {'learningRate': learningRate}

def get_adagrad_optimizer_params(globalStep, **kwargs):
    #learningRate = _get_learning_rate_piecewise_constant(globalStep, **kwargs)
    learningRate = _get_learning_rate_piecewise_shifted(globalStep, **kwargs)
    return {'learningRate': learningRate}

