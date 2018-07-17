"""
Created on 2018 7.2
@author: ShawnFu
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np  
import os
from PIL import Image
import cv2
import config
def predict(rob):
    output_graph_path = pb_name
    with tf.Session() as sess:
    # with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     sess.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        input_x = sess.graph.get_tensor_by_name("inputs:0")
        #is_training = sess.graph.get_tensor_by_name("is_training:0")
        output = sess.graph.get_tensor_by_name("outputs:0")
        #input_y = sess.graph.get_tensor_by_name("Placeholder_2:0")
    #rad = np.array(tf.random_normal([1,1,1,1000]))
        image = rob
        image = image / 255.0
        image = image - 0.5
        image = image * 2
        result = sess.run(output, feed_dict={input_x:np.reshape(image, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])})
        print(result)
        res=[]
        for i,x in enumerate(result[0]):
            res.append([i,x])
        res.sort(key=lambda y:y[1])
        
        print(labels[str(res[-1][0])])
        print(res[-1][1])
    sess.close()
