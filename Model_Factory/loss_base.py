from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops

def add_loss_summaries(total_loss, batchSize):
    """Add summaries for losses in calusa_heatmap model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    #loss_averages = tf.train.ExponentialMovingAverage(0.9, name='Average')
    losses = tf.get_collection('losses')
    #loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Individual average loss
#    lossPixelIndividual = tf.sqrt(tf.multiply(total_loss, 2/(batchSize*4))) # dvidied by (8/2) = 4 which is equal to sum of 2 of them then sqrt will result in euclidean pixel error
#    tf.summary.scalar('Average_Pixel_Error_Real', lossPixelIndividual)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
#    for l in losses + [total_loss]:
    for l in losses:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        #tf.summary.scalar(l.op.name, loss_averages.average(l))

    return

def _l2_loss(pred, tval): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    
    Returns:
      Loss tensor of type float.
    """
    #if not batch_size:
    #    batch_size = kwargs.get('train_batch_size')
    
    #l1_loss = tf.abs(tf.subtract(logits, HAB), name="abs_loss")
    #l1_loss_mean = tf.reduce_mean(l1_loss, name='abs_loss_mean')
    #tf.add_to_collection('losses', l2_loss_mean)

    l2_loss = tf.nn.l2_loss(tf.subtract(pred, tval), name="loss_l2")
    tf.add_to_collection('losses', l2_loss)

    #l2_loss_mean = tf.reduce_mean(l2_loss, name='l2_loss_mean')
    #tf.add_to_collection('losses', l2_loss_mean)

    #mse = tf.reduce_mean(tf.square(logits - HAB), name="mse")
    #tf.add_to_collection('losses', mse)

    # Calculate the average cross entropy loss across the batch.
    # labels = tf.cast(labels, tf.int64)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits, labels, name='cross_entropy_per_example')
    # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='loss_total')
####################################################################################

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=1):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-5, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)
#######################################################################################

def _weighted_L2_loss(tMatP, tMatT, activeBatchSize):
    mask = np.array([[100, 100, 100, 1, 100, 100, 100, 1, 100, 100, 100, 1]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    tMatP = tf.multiply(mask, tMatP)
    tMatT = tf.multiply(mask, tMatT)
    return _l2_loss(tMatP, tMatT)

def _weighted_params_L2_loss(targetP, targetT, activeBatchSize, outputSize=6, betareg=0, l2reg=0):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    ######## targetP, targetT shape Assertion
    targetP = tf.reshape(targetP, [int(targetP.get_shape()[0]), int(targetP.get_shape()[1])])
    targetT = tf.reshape(targetT, [int(targetT.get_shape()[0]), int(targetT.get_shape()[1])])

    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    if outputSize == 6: # euler parametric
        mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    elif outputSize == 7: #quaternion parametric
        #mask = np.array([[1000, 1000, 1000, 1000, 1, 1, 1]], dtype=np.float32)
        mask = np.array([[100, 1000000, 1000000, 1000000, 1, 1, 1]], dtype=np.float32)
    else:
        raise ValueError("Not [Euler, Quaternion] ouput!!! --- check the target output size and nature")
    #mask = np.array([[1000, 1000, 1000, 100, 100, 100]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return tf.add(_l2_loss(targetP, targetT), betareg*l2reg, name='loss_w_l2_params')

def _weighted_params_3t_L2_loss(targetP, targetT, activeBatchSize, outputSize=6, betareg=0, l2reg=0):
    '''
    Input:
        targetP: [batchsize, parametersize, tuplesize] #in this case tuplesize=2
        targetT: [batchsize, parametersize, tuplesize] #in this case tuplesize=2
    Output:
        [l2(pBA, tBA)] + [l2(pCB, tCB)] + [l2((pBA^2 + pCB^2), (tBA^2 + tCB^2))] + betareg*l2reg
    '''
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    ######## targetP, targetT shape Assertion

    # Importance weigting on angles as they have smaller values
    if outputSize == 6: # euler parametric
        mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    elif outputSize == 7: #quaternion parametric
        #mask = np.array([[1000, 1000, 1000, 1000, 1, 1, 1]], dtype=np.float32)
        mask = np.array([[100, 1000000, 1000000, 1000000, 1, 1, 1]], dtype=np.float32)
    else:
        raise ValueError("Not [Euler, Quaternion] ouput!!! --- check the target output size and nature")
    #mask = np.array([[1000, 1000, 1000, 100, 100, 100]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP[:, : , :, 0] = tf.multiply(targetP[:, :, :, 0], mask)
    targetP[:, : , :, 1] = tf.multiply(targetP[:, :, :, 1], mask)
    targetT[:, : , :, 0] = tf.multiply(targetT[:, :, :, 0], mask)
    targetT[:, : , :, 1] = tf.multiply(targetT[:, :, :, 1], mask)
    return tf.add(tf.add(tf.add(_l2_loss(targetP[:, :, :, 0], targetT[:, :, :, 0]), _l2_loss(targetP[:, :, :, 1], targetT[:, :, :, 1])),
                         _l2_loss(tf.add(targetP[:, :, :, 0], targetP[:, :, :, 1]), tf.add(targetT[:, :, :, 0], targetT[:, :, :, 1]))),
                  betareg*l2reg,
                  name='loss_w_3t_l2_params')

def _weighted_params_L2_loss_nTuple_last(targetP, targetT, nTuple, activeBatchSize):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return _l2_loss(targetP, targetT)

def _weighted_params_L2_loss_nTuple_all(targetP, targetT, nTuple, activeBatchSize):
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    mask = np.repeat(mask, nTuple-1, axis=0).reshape((nTuple-1)*6)
    mask = np.repeat(mask, activeBatchSize, axis=0)
    targetP = tf.multiply(targetP, mask)
    targetT = tf.multiply(targetT, mask)
    return _l2_loss(targetP, targetT)

def _params_classification_l2_loss_nTuple(targetP, targetT, nTuple, activeBatchSize):
    '''
    TargetT dimensions are [activeBatchSize, rows=6, cols=32, nTuple]
    '''
    # Alpha, Beta, Gamma are -Pi to Pi periodic radians - mod over pi to remove periodicity
    #mask = np.array([[np.pi, np.pi, np.pi, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, kwargs.get('activeBatchSize'), axis=0)
    #targetP = tf_mod(targetP, mask)
    #targetT = tf_mod(targetT, mask)
    # Importance weigting on angles as they have smaller values
    #mask = np.array([[100, 100, 100, 1, 1, 1]], dtype=np.float32)
    #mask = np.repeat(mask, nTuple-1, axis=0).reshape((nTuple-1)*6)
    #mask = np.repeat(mask, activeBatchSize, axis=0)
    #targetP = tf.multiply(targetP, mask)
    #targetT = tf.multiply(targetT, mask)
    targetT = tf.cast(targetT, tf.float32)
    return _l2_loss(targetP, targetT)

def _params_classification_softmaxCrossentropy_loss_nTuple(targetP, targetT, nTuple, activeBatchSize):
    '''
    Takes in the targetP and targetT and calculates the softmax-cross entropy loss for each parameter
    and sums them for each instance and sum again for each tuple in the batch
    TargetT dimensions are [activeBatchSize, rows=6, cols=32, nTuple]
    '''
    targetT = tf.cast(targetT, tf.float32)
    targetP = tf.cast(targetP, tf.float32)
    ############################
    # Alternatively, instead of sum, we could use squared_sum to penalize harsher
    ############################
    # ---> [activeBatchSize, rows=6, cols=32, nTuple]
    # Calculate softmax-cross entropy loss for each parameter (cols dimension -> cols)
    # ---> [activeBatchSize, rows=6, nTuple]
    # Then calculate sum of parameter losses for each batch (last 2 dimensions -> ntuple, rows), and returns an array of [activeBatchSize] size
    # ---> [activeBatchSize]
    smce_loss = tf.nn.l2_loss(tf.nn.softmax_cross_entropy_with_logits(logits=targetP, labels=targetT, dim=2), name="loss_smce_l2")
    #smce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=targetP, labels=targetT, dim=2), name="loss_smce_sum")
    return smce_loss

def _stixelnet_pl_loss(pred, tval, batchSize):
    '''
    Input:
        pred: prediction    --- batchSize x columns    (16 x 370)
        tval: ground truth  --- batchSize x columns    (16 x 370)
    Output:
    '''
    # softmax pred
    # then use precalculated centroid weights of bins to get the result
    # p_y = pred[i]*((c[i+1]-y)/(c[i+1]-c[i])) + pred[i+1]*((y-c[i])/(c[i+1]-c[i]))
    pY = tf.multiply(tf.nn.softmax(pred), tval)
    # sum per sample, take the log, negate the value, sum for all to get the loss
    # pl_loss = -np.log(p_y)
    loss_pl = tf.reduce_sum( tf.multiply( tf.log(tf.reduce_sum(pY, axis=1)),  -1.0 ),  name="loss_pl")    
    return loss_pl

def loss(pred, tval, **kwargs):
    """
    Choose the proper loss function and call it.
    """
    lossFunction = kwargs.get('lossFunction')
    if lossFunction == 'L2':
        return _l2_loss(pred, tval)
    if lossFunction == 'Weighted_L2_loss':
        return _weighted_L2_loss(pred, tval, kwargs.get('activeBatchSize'))
    if lossFunction == 'Weighted_Params_L2_loss':
        return _weighted_params_L2_loss(pred, tval, kwargs.get('activeBatchSize'))
    if lossFunction == 'Weighted_Params_L2_loss_nTuple_last':
        return _weighted_params_L2_loss_nTuple_last(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == 'Weighted_Params_L2_loss_nTuple_all':
        return _weighted_params_L2_loss_nTuple_all(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == '_params_classification_l2_loss_nTuple':
        return _params_classification_l2_loss_nTuple(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == '_params_classification_softmaxCrossentropy_loss_nTuple':
        if kwargs.get('lastTuple'):
            return _params_classification_softmaxCrossentropy_loss_nTuple(pred, tval, 1, kwargs.get('activeBatchSize'))
        else:
            return _params_classification_softmaxCrossentropy_loss_nTuple(pred, tval, kwargs.get('numTuple'), kwargs.get('activeBatchSize'))
    if lossFunction == 'stixelnet_pl_loss':
        return _stixelnet_pl_loss(pred, tval, kwargs.get('activeBatchSize'))
            





###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
######################### L2 WEIGHT RGULARIZATION #########################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

def _ohem_loss_l2reg(cls_prob, label, batchSize, beta, l2reg):
    '''
        cls_prob = batch * outputSize
        label = batch * outputSize
    '''
    lossInd = tf.nn.softmax_cross_entropy_with_logits_v2(logits=cls_prob, labels=label)
    lossInd,_ = tf.nn.top_k(lossInd, k=np.ceil(batchSize*0.5))
    lossInd = tf.add(tf.reduce_sum(lossInd), beta*l2reg, name="loss_ohem")
    return lossInd

def _clsf_smce_l2reg(targetP, targetT, beta, l2reg):
    '''
    Takes in the targetP and targetT and calculates the softmax-cross entropy loss for each parameter
    and sums them for each instance and sum again for each tuple in the batch
    TargetT dimensions are [activeBatchSize, rows]
    '''
    targetT = tf.cast(targetT, tf.float32)
    targetP = tf.cast(targetP, tf.float32)
    #smce_l2reg_loss = tf.add(tf.nn.l2_loss(tf.nn.softmax_cross_entropy_with_logits_v2(logits=targetP, labels=targetT, dim=1)), l2reg, name="loss_smce_l2_l2reg")
    smce_l2reg_loss = tf.add(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=targetP, labels=targetT, dim=1)), beta*l2reg, name="loss_smce_sum_l2reg")
    #smce_l2reg_loss = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=targetP, labels=targetT, dim=1)), beta*l2reg, name="loss_smce_mean_l2reg")
    return smce_l2reg_loss
    
def _focal_loss_l2reg(targetP, targetT, beta, l2reg, gamma=0.4): #gamma=0.4
    return tf.add(focal_loss(targetP, targetT, gamma=gamma), beta*l2reg, name="loss_focal")


def _stixelnet_pl_l2reg(pred, tval, beta, l2reg):
    '''
    Input:
        pred: prediction    --- batchSize x columns    (16 x 370)
        tval: ground truth  --- batchSize x columns    (16 x 370)
    Output:
    '''
    pY = tf.multiply(tf.log(tf.nn.softmax(pred)), tval)
    loss_pl = tf.reduce_sum(tf.multiply( tf.reduce_sum(pY, axis=1),  -1.0))
    return tf.add(loss_pl, beta*l2reg, name="loss_pl")

def loss_l2reg(pred, tval, l2reg, **kwargs):
    """
    Choose the proper loss function and call it.
    """
    lossFunction = kwargs.get('lossFunction')
    if lossFunction == 'Weighted_Params_L2_l2reg':
        return _weighted_params_L2_loss(pred, tval, kwargs.get('activeBatchSize'), kwargs.get('outputSize'), 0.01, l2reg)
    if lossFunction == 'Weighted_Params_3t_L2_l2reg':
        return _weighted_params_3t_L2_loss(pred, tval, kwargs.get('activeBatchSize'), kwargs.get('outputSize'), 0.01, l2reg)
    elif lossFunction == 'clsf_smce_l2reg':
        return _clsf_smce_l2reg(pred, tval, 0.1, l2reg) #0.01
    elif lossFunction == 'clsf_ohem_l2reg':
        return _ohem_loss_l2reg(pred, tval, kwargs.get('activeBatchSize'), 0, l2reg) #0.01
    elif lossFunction == 'clsf_focal_l2reg':
        return _focal_loss_l2reg(pred, tval, 0.1, l2reg) #0.01
    elif lossFunction == 'stixelnet_pl_l2reg':
        return _stixelnet_pl_l2reg(pred, tval, 0, l2reg)
