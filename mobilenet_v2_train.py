# -*- coding: utf-8 -*-
"""
Created on 2018 7.2
@author: ShawnFu
"""

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import argparse
import os
from PIL import Image
from datetime import datetime
import math
import time
import cv2
import config

from keras.utils import np_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from load_image import load_database_path, get_next_batch_from_path, shuffle_train_data
except:
    from load_image.load_image import load_database_path, get_next_batch_from_path, shuffle_train_data

# mobilenet_v2

from net.mobilenet_v2.mobilenet_v2 import mobilenet,training_scope

def arch_mobilenet_v2(X, num_classes, dropout_keep_prob=0.8, is_train=True):
    arg_scope = training_scope()
    with slim.arg_scope(arg_scope):
        net, end_points = mobilenet(X,num_classes=num_classes, is_training=is_train)
    return net
def train(train_data,train_label,valid_data,valid_label,train_n,valid_n,IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,batch_size=64,keep_prob=0.8,
           arch_model="arch_inception_v4"):

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    #Y = tf.placeholder(tf.float32, [None, 4])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    is_training = tf.placeholder(tf.bool, name='is_training')
    k_prob = tf.placeholder(tf.float32) # dropout

    # 定义模型

    net = arch_mobilenet_v2(X, num_classes, k_prob, is_training)

    # 
    #variables_to_restore,variables_to_train = g_parameter(checkpoint_exclude_scopes)

    # loss function
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = net))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = net))
    
    #var_list = variables_to_train
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gs=tf.Variable(0)
    lr = tf.train.exponential_decay(learning_rate=learning_rate, global_step=gs,decay_steps=2000,decay_rate=0.88)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9).minimize(loss, global_step=gs)
    predict = tf.reshape(net, [-1, num_classes])
    max_idx_p = tf.argmax(predict, 1)
    max_idx_l = tf.argmax(Y, 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # tensorboard
    with tf.name_scope('tmp/'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning rate',lr)
    summary_op = tf.summary.merge_all()
    #------------------------------------------------------------------------------------#
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    log_dir = arch_model + '_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    saver2 = tf.train.Saver(tf.global_variables())
    model_path = 'model/'

    ckpt = tf.train.get_checkpoint_state('model/')
    if ckpt and ckpt.model_checkpoint_path:
        saver2.restore(sess,ckpt.model_checkpoint_path)
        print('ckpt loaded')
    for epoch_i in range(epoch):
        for batch_i in range(int(train_n/batch_size)):
            images_train, labels_train = get_next_batch_from_path(train_data, train_label, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=True)
            los, _ = sess.run([loss,optimizer], feed_dict={X: images_train, Y: labels_train, k_prob:keep_prob, is_training:True})
            
            if batch_i % 100 == 0:
                images_valid, labels_valid = get_next_batch_from_path(valid_data, valid_label, batch_i%(int(valid_n/batch_size)), IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=False)
                ls, acc = sess.run([loss, accuracy], feed_dict={X: images_valid, Y: labels_valid, k_prob:1.0, is_training:False})
                print('Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, ls, acc))
                #if acc > 0.90:
                #    saver2.save(sess, model_path, global_step=batch_i, write_meta_graph=False)
            elif batch_i % 20 == 0:
                loss_, acc_, summary_str = sess.run([loss, accuracy, summary_op], feed_dict={X: images_train, Y: labels_train, k_prob:1.0, is_training:False})
                LR=sess.run(lr)
                writer.add_summary(summary_str, global_step=((int(train_n/batch_size))*epoch_i+batch_i))
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}, Learn rate: {:>3.10f}'.format(batch_i, loss_, acc_,LR))
                 
        print('Epoch===================================>: {:>2}'.format(epoch_i))
        valid_ls = 0
        valid_acc = 0
        for batch_i in range(int(valid_n/batch_size)):
            images_valid, labels_valid = get_next_batch_from_path(valid_data, valid_label, batch_i, IMAGE_HEIGHT, IMAGE_WIDTH, batch_size=batch_size, is_train=False)
            epoch_ls, epoch_acc = sess.run([loss, accuracy], feed_dict={X: images_valid, Y: labels_valid, k_prob:1.0, is_training:False})
            valid_ls = valid_ls + epoch_ls
            valid_acc = valid_acc + epoch_acc
        print('Epoch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(epoch_i, valid_ls/int(valid_n/batch_size), valid_acc/int(valid_n/batch_size)))
        if valid_acc/int(valid_n/batch_size) > 0.90:
            saver2.save(sess, model_path, global_step=epoch_i, write_meta_graph=True)
        
        print('>>>>>>>>>>>>>>>>>>>shuffle train_data<<<<<<<<<<<<<<<<<')
        # 每个epoch，重新打乱一次训练集：
        train_data, train_label = shuffle_train_data(train_data, train_label)
    writer.close()       
    sess.close()

if __name__ == '__main__':

    IMAGE_HEIGHT = config.IMAGE_HEIGHT
    IMAGE_WIDTH = config.IMAGE_WIDTH
    num_classes = config.num_classes
    epoch = config.epoch
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    keep_prob = config.keep_prob
    train_rate = config.train_rate
    craterDir = config.craterDir
    arch_model=config.arch_model
    checkpoint_exclude_scopes = config.checkpoint_exclude_scopes
    checkpoint_path=config.checkpoint_path
    print ("-----------------------------load_image.py start--------------------------")
    X_sample, Y_sample = load_database_path(craterDir)
    image_n = len(X_sample)
    print ("样本的总数量:")
    print (image_n)
    train_n = int(image_n*train_rate)
    valid_n = int(image_n*(1-train_rate))
    train_data, train_label = X_sample[0:train_n], Y_sample[0:train_n]
    valid_data, valid_label = X_sample[train_n:image_n], Y_sample[train_n:image_n]
    train_label = np_utils.to_categorical(train_label, num_classes)
    valid_label = np_utils.to_categorical(valid_label, num_classes)

    print ("-----------------------------train.py start--------------------------")
    train(train_data,train_label,valid_data,valid_label,train_n,valid_n,IMAGE_HEIGHT,IMAGE_WIDTH,learning_rate,num_classes,epoch,batch_size,keep_prob,
          arch_model)
