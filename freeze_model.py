import os
import tensorflow as tf

import numpy as np

import config
import argparse
slim = tf.contrib.slim

from net.mobilenet_v2.mobilenet_v2 import mobilenet,training_scope

dir = os.path.dirname(os.path.realpath(__file__))

def arch_mobilenet_v2(X, num_classes, dropout_keep_prob=0.8, is_train=True):
    arg_scope = training_scope()
    with slim.arg_scope(arg_scope):
        net, end_points = mobilenet(X,num_classes=num_classes, is_training=is_train)
    return net

def freeze_graph():
    # Load the args from the original experiment.
    IMAGE_HEIGHT = config.IMAGE_HEIGHT
    IMAGE_WIDTH = config.IMAGE_WIDTH
    num_classes = config.num_classes
    checkpoint_path=config.checkpoint_path


    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='inputs')
        Y = tf.placeholder(tf.float32, [None, num_classes])
        k_prob = tf.placeholder(tf.float32) # dropout
        is_training = tf.placeholder_with_default(False, [], 'is_training')

        # Create the model and an embedding head.
        net = arch_mobilenet_v2(X, num_classes, k_prob, is_training)

        predict = tf.reshape(net, [-1, num_classes])
        
        #outputs_node = tf.argmax(predict, 1, name='outputs')
        outputs_node = tf.nn.softmax(predict, name='outputs')


    output_graph = pb_name

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state('model/')
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            graph.as_graph_def(), # The graph_def is used to retrieve the nodes 
            #output_node_names.split(",") # The output node names are used to select the usefull nodes
            ['outputs']
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    #parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    #args = parser.parse_args()

    #freeze_graph(args.model_dir, args.output_node_names)
    freeze_graph()

