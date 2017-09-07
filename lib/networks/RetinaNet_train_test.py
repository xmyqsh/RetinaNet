# --------------------------------------------------------
# FPN - FPN
# Copyright (c) 2017
# Licensed under The MIT License [see LICENSE for details]
# Written by xmyqsh
# --------------------------------------------------------

import tensorflow as tf
import numpy as np
from .network import Network
from ..fast_rcnn.config import cfg


class RetinaNet_train_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

    def setup(self):

        n_classes = cfg.NCLASSES
        #pi = 0.01
        pi = 0.0001
        num_anchor_ratio = 3 # 1:2, 1:1, 2:1
        num_anchor_scale = 3 # 2**0, 2**(1/3), 2**(2/3)
        num_anchor = num_anchor_ratio * num_anchor_scale
        anchor_size = [None, None, None, 32, 64, 128, 256, 512] # P6 should be in RPN, but not Fast-RCNN, according to the paper
        _feat_stride = [None, 2, 4, 8, 16, 32, 64, 128]

        with tf.variable_scope('res1_2'):

            (self.feed('data')
                 .conv(7, 7, 64, 2, 2, relu=False, name='conv1')
                 .batch_normalization(relu=True, name='bn_conv1', is_training=False)
                 .max_pool(3, 3, 2, 2, padding='VALID',name='pool1')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
                 .batch_normalization(name='bn2a_branch1',is_training=False,relu=False))

            (self.feed('pool1')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
                 .batch_normalization(relu=True, name='bn2a_branch2a',is_training=False)
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
                 .batch_normalization(relu=True, name='bn2a_branch2b',is_training=False)
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
                 .batch_normalization(name='bn2a_branch2c',is_training=False,relu=False))

            (self.feed('bn2a_branch1',
                       'bn2a_branch2c')
                 .add(name='res2a')
                 .relu(name='res2a_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
                 .batch_normalization(relu=True, name='bn2b_branch2a',is_training=False)
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
                 .batch_normalization(relu=True, name='bn2b_branch2b',is_training=False)
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
                 .batch_normalization(name='bn2b_branch2c',is_training=False,relu=False))

            (self.feed('res2a_relu',
                       'bn2b_branch2c')
                 .add(name='res2b')
                 .relu(name='res2b_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
                 .batch_normalization(relu=True, name='bn2c_branch2a',is_training=False)
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
                 .batch_normalization(relu=True, name='bn2c_branch2b',is_training=False)
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
                 .batch_normalization(name='bn2c_branch2c',is_training=False,relu=False))

        with tf.variable_scope('res3_5'):

            (self.feed('res2b_relu',
                       'bn2c_branch2c')
                 .add(name='res2c')
                 .relu(name='res2c_relu')
                 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1', padding='VALID')
                 .batch_normalization(name='bn3a_branch1',is_training=False,relu=False))

            (self.feed('res2c_relu')
                 .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a', padding='VALID')
                 .batch_normalization(relu=True, name='bn3a_branch2a',is_training=False)
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
                 .batch_normalization(relu=True, name='bn3a_branch2b',is_training=False)
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
                 .batch_normalization(name='bn3a_branch2c',is_training=False,relu=False))

            (self.feed('bn3a_branch1',
                       'bn3a_branch2c')
                 .add(name='res3a')
                 .relu(name='res3a_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
                 .batch_normalization(relu=True, name='bn3b_branch2a',is_training=False)
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
                 .batch_normalization(relu=True, name='bn3b_branch2b',is_training=False)
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
                 .batch_normalization(name='bn3b_branch2c',is_training=False,relu=False))

            (self.feed('res3a_relu',
                       'bn3b_branch2c')
                 .add(name='res3b')
                 .relu(name='res3b_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
                 .batch_normalization(relu=True, name='bn3c_branch2a',is_training=False)
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
                 .batch_normalization(relu=True, name='bn3c_branch2b',is_training=False)
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
                 .batch_normalization(name='bn3c_branch2c',is_training=False,relu=False))

            (self.feed('res3b_relu',
                       'bn3c_branch2c')
                 .add(name='res3c')
                 .relu(name='res3c_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
                 .batch_normalization(relu=True, name='bn3d_branch2a',is_training=False)
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
                 .batch_normalization(relu=True, name='bn3d_branch2b',is_training=False)
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
                 .batch_normalization(name='bn3d_branch2c',is_training=False,relu=False))

            (self.feed('res3c_relu',
                       'bn3d_branch2c')
                 .add(name='res3d')
                 .relu(name='res3d_relu')
                 .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1', padding='VALID')
                 .batch_normalization(name='bn4a_branch1',is_training=False,relu=False))

            (self.feed('res3d_relu')
                 .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a', padding='VALID')
                 .batch_normalization(relu=True, name='bn4a_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
                 .batch_normalization(relu=True, name='bn4a_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
                 .batch_normalization(name='bn4a_branch2c',is_training=False,relu=False))

            (self.feed('bn4a_branch1',
                       'bn4a_branch2c')
                 .add(name='res4a')
                 .relu(name='res4a_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
                 .batch_normalization(relu=True, name='bn4b_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
                 .batch_normalization(relu=True, name='bn4b_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
                 .batch_normalization(name='bn4b_branch2c',is_training=False,relu=False))

            (self.feed('res4a_relu',
                       'bn4b_branch2c')
                 .add(name='res4b')
                 .relu(name='res4b_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
                 .batch_normalization(relu=True, name='bn4c_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4c_branch2b')
                 .batch_normalization(relu=True, name='bn4c_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c')
                 .batch_normalization(name='bn4c_branch2c',is_training=False,relu=False))

            (self.feed('res4b_relu',
                       'bn4c_branch2c')
                 .add(name='res4c')
                 .relu(name='res4c_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
                 .batch_normalization(relu=True, name='bn4d_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4d_branch2b')
                 .batch_normalization(relu=True, name='bn4d_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c')
                 .batch_normalization(name='bn4d_branch2c',is_training=False,relu=False))

            (self.feed('res4c_relu',
                       'bn4d_branch2c')
                 .add(name='res4d')
                 .relu(name='res4d_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
                 .batch_normalization(relu=True, name='bn4e_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4e_branch2b')
                 .batch_normalization(relu=True, name='bn4e_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c')
                 .batch_normalization(name='bn4e_branch2c',is_training=False,relu=False))

            (self.feed('res4d_relu',
                       'bn4e_branch2c')
                 .add(name='res4e')
                 .relu(name='res4e_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
                 .batch_normalization(relu=True, name='bn4f_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4f_branch2b')
                 .batch_normalization(relu=True, name='bn4f_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c')
                 .batch_normalization(name='bn4f_branch2c',is_training=False,relu=False))

            (self.feed('res4e_relu',
                       'bn4f_branch2c')
                 .add(name='res4f')
                 .relu(name='res4f_relu'))

            # conv5
            (self.feed('res4f_relu')
                 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a', padding='VALID')
                 .batch_normalization(relu=True, name='bn5a_branch2a',is_training=False)
                 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
                 .batch_normalization(relu=True, name='bn5a_branch2b',is_training=False)
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
                 .batch_normalization(name='bn5a_branch2c',is_training=False,relu=False))

            (self.feed('res4f_relu')
                 .conv(1,1,2048,2,2,biased=False, relu=False, name='res5a_branch1', padding='VALID')
                 .batch_normalization(name='bn5a_branch1',is_training=False,relu=False))

            (self.feed('bn5a_branch2c','bn5a_branch1')
                 .add(name='res5a')
                 .relu(name='res5a_relu')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
                 .batch_normalization(relu=True, name='bn5b_branch2a',is_training=False)
                 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
                 .batch_normalization(relu=True, name='bn5b_branch2b',is_training=False)
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
                 .batch_normalization(name='bn5b_branch2c',is_training=False,relu=False))

            (self.feed('res5a_relu',
                       'bn5b_branch2c')
                 .add(name='res5b')
                 .relu(name='res5b_relu')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
                 .batch_normalization(relu=True, name='bn5c_branch2a',is_training=False)
                 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b')
                 .batch_normalization(relu=True, name='bn5c_branch2b',is_training=False)
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
                 .batch_normalization(name='bn5c_branch2c',is_training=False,relu=False))

            (self.feed('res5b_relu',
                       'bn5c_branch2c')
                 .add(name='res5c')
                 .relu(name='res5c_relu'))

        with tf.variable_scope('Top-Down'):

            # Top-Down
            (self.feed('res5c_relu') # C5
                 .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='P5'))

            (self.feed('res5c_relu') # C5
                 .conv(3, 3, 256, 2, 2, biased=True, relu=False, name='P6'))

            (self.feed('P6')
                 .relu(name='P6_relu')
                 .conv(3, 3, 256, 2, 2, biased=True, relu=False, name='P7'))

            (self.feed('res4f_relu') # C4
                 .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='C4_lateral'))

            (self.feed('P5',
                       'C4_lateral')
                 .upbilinear(name='C5_topdown'))

            (self.feed('C5_topdown',
                       'C4_lateral')
                 .add(name='P4_pre')
                 .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='P4'))

            (self.feed('res3d_relu') #C3
                 .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='C3_lateral'))

            (self.feed('P4',
                       'C3_lateral')
                 .upbilinear(name='C4_topdown'))

            (self.feed('C4_topdown',
                       'C3_lateral')
                 .add(name='P3_pre')
                 .conv(3, 3, 256, 1, 1, biased=True, relu= False, name='P3'))


        with tf.variable_scope('clsSubNet') as scope:

            (self.feed('P3')
                 .conv(3,3,256,1,1,name='cls_conv_1/P3',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_2/P3',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_3/P3',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_4/P3',reuse=True)
                 .conv(3,3,num_anchor*n_classes,1,1,cls_final=True,bias_init=-np.log((1-pi)/pi),relu=False,name='cls_score/P3',reuse=True)
                 #.conv(3,3,num_anchor*n_classes,1,1,relu=False,name='cls_score/P3',reuse=True)
                 .reshape_layer([-1, n_classes],name='cls_score_reshape/P3')
                 .softmax(name='cls_prob/P3'))

            scope.reuse_variables()

            (self.feed('P4')
                 .conv(3,3,256,1,1,name='cls_conv_1/P4',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_2/P4',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_3/P4',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_4/P4',reuse=True)
                 .conv(3,3,num_anchor*n_classes,1,1,cls_final=True,bias_init=-np.log((1-pi)/pi),relu=False,name='cls_score/P4',reuse=True)
                 #.conv(3,3,num_anchor*n_classes,1,1,relu=False,name='cls_score/P4',reuse=True)
                 .reshape_layer([-1, n_classes],name='cls_score_reshape/P4')
                 .softmax(name='cls_prob/P4'))

            (self.feed('P5')
                 .conv(3,3,256,1,1,name='cls_conv_1/P5',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_2/P5',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_3/P5',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_4/P5',reuse=True)
                 .conv(3,3,num_anchor*n_classes,1,1,cls_final=True,bias_init=-np.log((1-pi)/pi),relu=False,name='cls_score/P5',reuse=True)
                 #.conv(3,3,num_anchor*n_classes,1,1,relu=False,name='cls_score/P5',reuse=True)
                 .reshape_layer([-1, n_classes],name='cls_score_reshape/P5')
                 .softmax(name='cls_prob/P5'))

            (self.feed('P6')
                 .conv(3,3,256,1,1,name='cls_conv_1/P6',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_2/P6',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_3/P6',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_4/P6',reuse=True)
                 .conv(3,3,num_anchor*n_classes,1,1,cls_final=True,bias_init=-np.log((1-pi)/pi),relu=False,name='cls_score/P6',reuse=True)
                 #.conv(3,3,num_anchor*n_classes,1,1,relu=False,name='cls_score/P6',reuse=True)
                 .reshape_layer([-1, n_classes],name='cls_score_reshape/P6')
                 .softmax(name='cls_prob/P6'))

            (self.feed('P7')
                 .conv(3,3,256,1,1,name='cls_conv_1/P7',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_2/P7',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_3/P7',reuse=True)
                 .conv(3,3,256,1,1,name='cls_conv_4/P7',reuse=True)
                 .conv(3,3,num_anchor*n_classes,1,1,cls_final=True,bias_init=-np.log((1-pi)/pi),relu=False,name='cls_score/P7',reuse=True)
                 #.conv(3,3,num_anchor*n_classes,1,1,relu=False,name='cls_score/P7',reuse=True)
                 .reshape_layer([-1, n_classes],name='cls_score_reshape/P7')
                 .softmax(name='cls_prob/P7'))

            (self.feed('cls_score_reshape/P3',
                       'cls_score_reshape/P4',
                       'cls_score_reshape/P5',
                       'cls_score_reshape/P6',
                       'cls_score_reshape/P7')
                 .concat(0, name = 'cls_score_reshape_concat'))


        with tf.variable_scope('boxSubNet') as scope:

            (self.feed('P3')
                 .conv(3,3,256,1,1,name='box_conv_1/P3',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_2/P3',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_3/P3',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_4/P3',reuse=True)
                 .conv(3,3,num_anchor*4,1,1,relu=False,name='box_pred/P3',reuse=True)
                 .reshape_layer([-1, 4],name='box_pred_reshape/P3'))

            scope.reuse_variables()

            (self.feed('P4')
                 .conv(3,3,256,1,1,name='box_conv_1/P4',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_2/P4',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_3/P4',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_4/P4',reuse=True)
                 .conv(3,3,num_anchor*4,1,1,relu=False,name='box_pred/P4',reuse=True)
                 .reshape_layer([-1, 4],name='box_pred_reshape/P4'))

            (self.feed('P5')
                 .conv(3,3,256,1,1,name='box_conv_1/P5',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_2/P5',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_3/P5',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_4/P5',reuse=True)
                 .conv(3,3,num_anchor*4,1,1,relu=False,name='box_pred/P5',reuse=True)
                 .reshape_layer([-1, 4],name='box_pred_reshape/P5'))

            (self.feed('P6')
                 .conv(3,3,256,1,1,name='box_conv_1/P6',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_2/P6',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_3/P6',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_4/P6',reuse=True)
                 .conv(3,3,num_anchor*4,1,1,relu=False,name='box_pred/P6',reuse=True)
                 .reshape_layer([-1, 4],name='box_pred_reshape/P6'))

            (self.feed('P7')
                 .conv(3,3,256,1,1,name='box_conv_1/P7',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_2/P7',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_3/P7',reuse=True)
                 .conv(3,3,256,1,1,name='box_conv_4/P7',reuse=True)
                 .conv(3,3,num_anchor*4,1,1,relu=False,name='box_pred/P7',reuse=True)
                 .reshape_layer([-1, 4],name='box_pred_reshape/P7'))

            (self.feed('box_pred_reshape/P3',
                       'box_pred_reshape/P4',
                       'box_pred_reshape/P5',
                       'box_pred_reshape/P6',
                       'box_pred_reshape/P7')
                 .concat(0, name = 'box_pred_reshape_concat'))

            (self.feed('cls_score/P3',
                       'cls_score/P4',
                       'cls_score/P5',
                       'cls_score/P6',
                       'cls_score/P7',
                       'gt_boxes', 'im_info')
                 .anchor_target_layer(num_anchor_scale, _feat_stride[3:], anchor_size[3:], name = 'rpn-data'))

# ----------------------------------------------------------------------------------------------------------------------------

    def build_loss(self):
        ############# RPN
        cls_score = self.get_output('cls_score_reshape_concat') # shape(sum(HxWxA), 2)
        bbox_pred = self.get_output('box_pred_reshape_concat') # shape (sum(HxWxA), 4)

        label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)
        bbox_targets = self.get_output('rpn-data')[1]
        bbox_inside_weights = self.get_output('rpn-data')[2]
        bbox_outside_weights = self.get_output('rpn-data')[3]

        #fg_keep = tf.equal(label, 1)
        fg_keep = tf.where(tf.greater(label, 0))
        keep = tf.where(tf.not_equal(label, -1))

        #cls_score = tf.reshape(tf.gather(cls_score, keep), [-1, n_classes]) # shape (N, n_classes)
        cls_score = tf.reshape(tf.gather(cls_score, keep), [-1, 21]) # shape (N, n_classes)
        #bbox_pred = tf.reshape(tf.gather(bbox_pred, keep), [-1, 4]) # shape (N, 4)
        bbox_pred = tf.reshape(tf.gather(bbox_pred, fg_keep), [-1, 4]) # shape (N, 4)

        label = tf.reshape(tf.gather(label, keep), [-1])
        bbox_targets = tf.reshape(tf.gather(tf.reshape(bbox_targets, [-1, 4]), fg_keep), [-1, 4])
        bbox_inside_weights = tf.reshape(tf.gather(tf.reshape(bbox_inside_weights, [-1, 4]), fg_keep), [-1, 4])
        bbox_outside_weights = tf.reshape(tf.gather(tf.reshape(bbox_outside_weights, [-1, 4]), fg_keep), [-1, 4])
        '''
        bbox_targets = tf.reshape(tf.gather(tf.reshape(bbox_targets, [-1, 4]), keep), [-1, 4])
        bbox_inside_weights = tf.reshape(tf.gather(tf.reshape(bbox_inside_weights, [-1, 4]), keep), [-1, 4])
        bbox_outside_weights = tf.reshape(tf.gather(tf.reshape(bbox_outside_weights, [-1, 4]), keep), [-1, 4])
        '''

        #cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)
        #cross_entropy = tf.reduce_mean(cross_entropy_n)
        cls_prob = tf.nn.softmax(logits=cls_score)
        focal_loss_n = self.focal_loss(cls_prob, label, num_class=cfg.NCLASSES)
        #cross_entropy = tf.reduce_mean(focal_loss_n)
        cross_entropy = tf.reduce_sum(focal_loss_n) / (tf.reduce_sum(tf.cast(tf.greater(label, 0), tf.float32)) + 1.0)


        loss_box_n = tf.reduce_sum(self.smooth_l1_dist(
            bbox_pred - bbox_targets), axis=[1])
            #bbox_inside_weights * (bbox_pred - bbox_targets)), axis=[1])
        #loss_box = tf.reduce_sum(loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1.0)
        loss_box = tf.reduce_sum(loss_box_n) / (tf.reduce_sum(tf.cast(tf.greater(label, 0), tf.float32)) + 1.0)

        loss = cross_entropy + loss_box

        vs_name = ['res3_5', 'Top-Down', 'clsSubNet', 'boxSubNet']

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            print 'regularization_losses: '
            print regularization_losses
            regularization_losses = [v for v in regularization_losses for vs in vs_name if v.op.name.startswith(vs + '/')]
            print 'filtered regularization_losses: '
            print regularization_losses
            loss = tf.add_n(regularization_losses) + loss

        return loss, cross_entropy, loss_box
