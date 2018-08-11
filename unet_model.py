# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import argparse
#import cv2
parser = argparse.ArgumentParser()

# Basic Model Arguments/Parameters
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'Number of examples to process in a batch.')
parser.add_argument('--use_fp16', type = bool, default = False,
                    help = 'Train model using float16 data type.')

PARAMS = parser.parse_args()

WEIGHT_DECAY_LAMBDA = 0.000
WEIGHT_DECAY_COLLECTION = "my_losses"

def VariableSummary(var, name):
    """Add a lot of summaries to a tensor."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def XavierInitializer(prev_units, curr_units, kernel_size, stddev_factor=1.0):
    """Initialization for CONV2D in the style of Xavier Glorot et al.(2010).
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs.
    ArgS:
        prev_units: The number of channels in the previous layer.
        curr_units: The number of channels in the current layer.
        stddev_factor:
    Returns:
        Initial value of the weights of the current conv/transpose conv layer.
    """
    stddev = np.sqrt(stddev_factor / (np.sqrt(prev_units * curr_units) * kernel_size * kernel_size))

    return tf.truncated_normal_initializer(mean=0.0, stddev=stddev)

def VariableOnCPU(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if PARAMS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer = initializer, dtype = dtype)
    return var


def VariableWithWeightDecay(name, shape, wd, is_conv):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        init_mode: Initialization mode for variables: 'conv' and 'convT'

    Returns:
        Variable Tensor
    """
    if is_conv == True:
        initializer = XavierInitializer(shape[2], shape[3], shape[0], stddev_factor=1.0)
    else:
        initializer = XavierInitializer(shape[3], shape[2], shape[0], stddev_factor=1.0)

    var = VariableOnCPU(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection(WEIGHT_DECAY_COLLECTION, weight_decay)
    return var


def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def conv_layer(inputs, kernel_shape, wd, name):
    with tf.variable_scope(name):
        W = VariableWithWeightDecay('weights', shape=kernel_shape, wd=wd, is_conv=True)
        b = VariableOnCPU('biases', [kernel_shape[3]], tf.constant_initializer())
        conv = tf.nn.relu(conv2d(inputs, W, name='conv_op') + b, name='relu_op')

    return conv


def norm_layer(inputs, name):
    with tf.variable_scope(name):
        norm = tf.nn.lrn(inputs, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
        VariableSummary(norm)

    return norm


def conv2d_T(x, W, output_shape, name):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME', name=name)


def convT_layer(inputs, kernel_shape, output_shape, wd, name):
    with tf.variable_scope(name):
        W = VariableWithWeightDecay('weights_T', shape=kernel_shape, wd=wd, is_conv=False)
        b = VariableOnCPU('biases_T', [kernel_shape[2]], tf.constant_initializer())
        convT = tf.nn.relu(conv2d_T(inputs, W, output_shape=output_shape, name='conv_T') + b, name='relu_op')

    return convT


def max_pooling_2x2(x, name):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


def UNet(inputs):
    ''' build the model
    Args:
        GIBBS_images : GIBBS images with size of [batch_size, GIBBS_W, GIBBS_H, Channels]
    Returns:
        predicted_CLEAR: a tensor with size [batch_size, CLEAR_W, CLEAR_H, Channels]
    '''

    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    # inputs: [BATCH_SIZE, 184, 184, 1]

    in_maps = int(inputs.get_shape()[3])
    kernel_size = 3
    
    inputs = conv_layer(inputs, [kernel_size, kernel_size, in_maps, 64],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer1')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 64],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer2')
    conv2 = inputs
    layer2_shape = tf.shape(inputs)
    #[1, 184, 184, 64]
    
    # pool3: [BATCH_SIZE, 184, 184, 64]
    #        [BATCH_SZIE, 92, 92, 64]
    inputs = max_pooling_2x2(inputs, 'pool_layer3')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 128],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer4')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 128],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer5')
    conv5 = inputs  #[1, 92, 92, 128]
    layer5_shape = tf.shape(inputs)


    # pool6: [BATCH_SIZE, 92, 92, 128]
    #        [BATCH_SZIE, 46, 46, 128]
    inputs = max_pooling_2x2(inputs, 'pool_layer6')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 256],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer7')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 256],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer8')
    conv8 = inputs  #[1, 64, 64, 256]
    layer8_shape = tf.shape(inputs)


    # pool9: [BATCH_SIZE, 46, 46, 256]
    #        [BATCH_SZIE, 23, 23, 256]
    inputs = max_pooling_2x2(inputs, 'pool_layer9')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer10')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer11')
    conv11 = inputs     #[1, 23, 23, 512]
    layer11_shape = tf.shape(inputs)

    
    # pool12: [BATCH_SIZE, 23, 23, 512]
    #         [BATCH_SZIE, 12, 12, 512]
    inputs = max_pooling_2x2(inputs, 'pool_layer12')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 1024],  #[1, 12, 12, 1024]
                        WEIGHT_DECAY_LAMBDA, 'conv_layer13')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 1024, 1024], #[1, 12, 12, 1024]
                        WEIGHT_DECAY_LAMBDA, 'conv_layer14')


    # convT15: [BATCH_SIZE, 12, 12, 1024]
    #          [BATCH_SZIE, 12, 12, 512]
    inputs = convT_layer(inputs, [kernel_size, kernel_size, 512, 1024], layer11_shape,
                         WEIGHT_DECAY_LAMBDA, 'convT_layer15')      #[1, 23, 23, 512]
    inputs = conv_layer(tf.concat([inputs, conv11], 3),[kernel_size, kernel_size, 1024, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer16')        #[1, 23, 23, 512]
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer17')


    # convT18: [BATCH_SIZE, 23, 23, 512]
    #          [BATCH_SZIE, 12, 12, 256]
    inputs = convT_layer(inputs, [kernel_size, kernel_size, 256, 512], layer8_shape,    #[1, 46, 46, 256]
                         WEIGHT_DECAY_LAMBDA, 'convT_layer18')
    inputs = conv_layer(tf.concat([inputs, conv8], 3), [kernel_size, kernel_size, 512, 256],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer19')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 256],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer20')

    
    # convT21: [BATCH_SIZE, 46, 46, 256]
    #          [BATCH_SZIE, 24, 24, 128]
    inputs = convT_layer(inputs, [kernel_size, kernel_size, 128, 256], layer5_shape,    #[1, 92, 92, 128]
                          WEIGHT_DECAY_LAMBDA, 'convT_layer21')
    inputs = conv_layer(tf.concat([inputs, conv5], 3), [kernel_size, kernel_size, 256, 128],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer22')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 128],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer23')


    # convT24: [BATCH_SIZE, 92, 92, 128]
    #
    inputs = convT_layer(inputs, [kernel_size, kernel_size, 64, 128], layer2_shape, #[1, 184, 184, 64]
                          WEIGHT_DECAY_LAMBDA,'convT_layer24')
    inputs = conv_layer(tf.concat([inputs, conv2], 3), [kernel_size, kernel_size, 128, 64],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer25')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 64],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer26')
    
    inputs = conv_layer(inputs, [1, 1, 64, 1], WEIGHT_DECAY_LAMBDA, 'conv_layer27')
    
    # Filter exception value
    inputs = tf.minimum(tf.maximum(inputs, 0.0), 1.0)
    #inputs = tf.clip_by_value(inputs, 0.0, 1.0)

    return inputs


def loss(labels, pred, lamda = 10):
    """Calculates the loss from the real seg images and the predicted images.

    Args:
        CLEAR: the corresponding true CLEAR images, i.e. reference CLEAR images => [b, h, w, c]
        pred_CLEAR: predicted CLEAR images by the model.                     => [b, h, w, c]

    Returns:
        loss: MSE between real CLEAR images and predicted CLEAR images.
    """
    with tf.name_scope('loss'):
#        pred = tf.to_float(pred > 0)
#        labels = tf.to_float(labels > 0)
#        intersection = tf.reduce_sum(pred * labels)
#        summation = tf.reduce_sum(pred) + tf.reduce_sum(labels)
#        loss_dice = 1. - (2.0 * intersection + lamda)/(summation + lamda)

        mse_loss = tf.losses.mean_squared_error(labels, pred)
        tf.add_to_collection(WEIGHT_DECAY_COLLECTION, mse_loss)
        tf.summary.scalar(mse_loss.op.name, mse_loss)

        total_losses = tf.add_n(tf.get_collection(WEIGHT_DECAY_COLLECTION))

    return total_losses

# def loss(labels, pred, lamda = 0.001, TV = True):
#     """Calculates the loss from the real seg images and the predicted images.
#
#     Args:
#         CLEAR: the corresponding true CLEAR images, i.e. reference CLEAR images => [b, h, w, c]
#         pred_CLEAR: predicted CLEAR images by the model.                     => [b, h, w, c]
#
#     Returns:
#         loss: MSE between real CLEAR images and predicted CLEAR images.
#     """
#     with tf.name_scope('loss'):
#
#         ###########Dice##########
#         labels = tf.to_float(labels > 0.)
#         pred = tf.to_float(pred > 0.)
#
#         #axes = [1, 2, 3]
#         intersection = tf.reduce_sum(labels * pred)
#         summation = tf.reduce_sum(labels) + tf.reduce_sum(pred)
#
#         loss_dice =1. - (2.0 * intersection + lamda) / (summation + lamda)
#
#         tf.add_to_collection(WEIGHT_DECAY_COLLECTION, loss_dice)
#         tf.summary.scalar(loss_dice.op.name, loss_dice)
#         total_losses = tf.reduce_mean(tf.add_n(tf.get_collection(WEIGHT_DECAY_COLLECTION)))
#
#         return total_losses
#

def optimize(loss, learning_rate, global_step):
    """Sets up the training Ops.
    
    Args:
        loss: Loss tensor, from loss().
        lr: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    decayed_lr = tf.train.exponential_decay(learning_rate,
                                            global_step,
                                            decay_steps = 500,
                                            decay_rate = 0.9,
                                            staircase = True)
    with tf.name_scope('learning_rate'):
        tf.summary.scalar('learning_rate', decayed_lr)
        tf.summary.histogram('histogram', decayed_lr)
        
    with tf.name_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate = decayed_lr)
        
        # with a global_step to track the global step.
        train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op


def evaluation(CLEAR, pred_CLEAR):
    """Evaluate the quality of the predicted CLEAR images at predicting the CLEAR iamges.

    Args:
        CLEAR: The real CLEAR images.
        pred_CLEAR: Predicted CLEAR images by the model.

    Returns:
        mPSNR: mean PSNR between the real CLEAR images and predicted CLEAR images.
    """
    with tf.name_scope('psnr'):
        MSE = tf.reduce_mean(tf.square(CLEAR - pred_CLEAR), [1, 2, 3])
        T = tf.div(1.0, MSE)  # 除法
        R = 10.0 * tf.div(tf.log(T), tf.log(10.0))
        mPSNR = tf.reduce_mean(R, name='mean_psnr')

    # Attach a scalar summary to mPSNR
    tf.summary.scalar(mPSNR.op.name, mPSNR)

    return mPSNR


def Dice(img_label, img_seg):
    '''
   Compute the dics  coefficient between the label images and the predicted label images.
   Before dics is a Threshold operation.

   Args:
       img_label_batch: ground truth label images.
       img_seg_batch: predicted label images.

   Returns: dics coefficient.
'''
    labels = tf.to_float(img_label > 0.)
    pred = tf.to_float(img_seg > 0.)

    axes = (1, 2, 3)
    intersection = tf.reduce_sum(labels * pred, axis=axes)
    summation = tf.reduce_sum(labels, axis=axes) + tf.reduce_sum(pred, axis=axes)

    dice = tf.reduce_mean((2.0 * intersection + 0.01) / (summation + 0.01))

    return dice











