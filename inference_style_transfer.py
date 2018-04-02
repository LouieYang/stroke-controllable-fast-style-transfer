from __future__ import division
import os
from os import listdir
from os.path import isfile, join
import time
import tensorflow as tf
import numpy as np
import sys
import functools
from scipy.misc import imresize
from utils import *
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model', required=True)
parser.add_argument('--serial', dest='serial', default='./examples/serial/default')
parser.add_argument('--content', dest='content', required=True)
parser.add_argument('--interp', dest='interp', type=int, default=-1)

args = parser.parse_args()

STROKE_SHORTCUT_DICT = {"768": [False, False], "512": [False, True], "256": [True, False], "interp": [True, True]}

with open(args.model, 'rb') as f:
    style_graph_def = tf.GraphDef()
    style_graph_def.ParseFromString(f.read())

style_graph = tf.Graph()
with style_graph.as_default():
    tf.import_graph_def(style_graph_def, name='')
style_graph.finalize()

sess_style = tf.Session(graph = style_graph)
content_tensor = style_graph.get_tensor_by_name('content_input:0')
shortcut_options = style_graph.get_tensor_by_name('shortcut:0')
interp_options = style_graph.get_tensor_by_name('interpolation_factor:0')
style_output_tensor = style_graph.get_tensor_by_name('add_39:0')

# TODO: remove here by deleting training batch size dependencies
train_batch_size = content_tensor.get_shape().as_list()[0]

img = np.array(load_image(args.content, 1024), dtype=np.float32)
border = np.ceil(np.shape(img)[0]/20/4).astype(int) * 5
container = [imresize(img, (np.shape(img)[0] + 2 * border, np.shape(img)[1] + 2 * border, 3))]
container[0][border : np.shape(img)[0] + border, border : np.shape(img)[1] + border, :] = img
container = np.repeat(container, train_batch_size, 0)

mkdir_if_not_exists(args.serial)
if args.interp < 0:
    style_768 = sess_style.run(style_output_tensor, feed_dict={content_tensor: container, shortcut_options: STROKE_SHORTCUT_DICT["768"], interp_options: 0})
    style_512 = sess_style.run(style_output_tensor, feed_dict={content_tensor: container, shortcut_options: STROKE_SHORTCUT_DICT["512"], interp_options: 0})
    style_256 = sess_style.run(style_output_tensor, feed_dict={content_tensor: container, shortcut_options: STROKE_SHORTCUT_DICT["256"], interp_options: 0})
    
    save_image(os.path.join(args.serial, "style_768.jpg"), np.squeeze(style_768[0][border : np.shape(img)[0] + border, border : np.shape(img)[1] + border, :]))
    save_image(os.path.join(args.serial, "style_512.jpg"), np.squeeze(style_512[0][border : np.shape(img)[0] + border, border : np.shape(img)[1] + border, :]))
    save_image(os.path.join(args.serial, "style_256.jpg"), np.squeeze(style_256[0][border : np.shape(img)[0] + border, border : np.shape(img)[1] + border, :]))
else:
    for i in xrange(args.interp):
        style_img = sess_style.run(
            style_output_tensor,
            feed_dict={
                content_tensor: container, 
                shortcut_options: STROKE_SHORTCUT_DICT["interp"], 
                interp_options: i / args.interp * 2
            })
        save_image(
            os.path.join(args.serial, "style_interp_{}_{}.jpg".format(i, args.interp)),
            np.squeeze(style_img[0][border : np.shape(img)[0] + border, border : np.shape(img)[1] + border, :])
            )

sess_style.close()
   