from __future__ import division
import os
from os import listdir
from os.path import isfile, join
import time
import tensorflow as tf
import numpy as np
import sys
from scipy.misc import imresize
import functools

import vgg19.vgg as vgg
from utils import *
import netdef

STROKE_SHORTCUT_DICT = {"768": [False, False], "512": [False, True], "256": [True, False], "interp": [True, True]}
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')
CONTENT_LAYERS = ('conv4_2')

DEFAULT_RESOLUTIONS = ((768, 768), (512, 512), (256, 256))

class DataLoader(object):
    def __init__(self, args):
        file_names = [join(args.train_path, f) for f in listdir(args.train_path) if isfile(join(args.train_path, f)) and ".jpg" in f]
        self.mscoco_fnames = file_names
        self.train_size = len(file_names)
        self.batch_size = args.batch_size
        self.epochs = 0
        self.nbatches = int(self.train_size / args.batch_size)
        self.batch_idx = 0
        self.perm = np.random.permutation(self.train_size)

        print ("[*] Training dataset size: {}".format(self.train_size))
        print ("[*] Batch size: {}".format(self.batch_size))
        print ("[*] {} #Batches per epoch".format(self.nbatches))

    def fill_feed_dict(self, content_pl, img_size=None):
        content_images = np.zeros((self.batch_size,) + img_size, dtype=np.float32)
        for i in xrange(self.batch_size):
            img = np.array(load_image(self.mscoco_fnames[self.perm[self.batch_idx * self.batch_size + i]], shape=img_size), dtype=np.float32)
            content_images[i] = img

        self.batch_idx += 1
        if self.batch_idx == self.nbatches:
            self.batch_idx = 0
            self.epochs += 1
            self.perm = np.random.permutation(self.train_size)

        return {content_pl: content_images}

class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self._build_model(args)
        self.saver = tf.train.Saver(max_to_keep=None)

        self.data_loader = DataLoader(args)

    def _build_model(self, args):
        # center-crop loading style image
        # change this the following two lines to load original style image
        style_highres_img = load_image(args.style, shape=DEFAULT_RESOLUTIONS[1])
        self.style_targets = [np.array(style_highres_img.resize((shape[0], shape[1]), resample=Image.BILINEAR), dtype=np.float32)
                                for shape in DEFAULT_RESOLUTIONS]

        self.content_input = tf.placeholder(tf.float32, shape=(args.batch_size, None, None, 3), name='content_input')
        self.shortcut = tf.placeholder_with_default([False, False], shape=[2], name="shortcut")
        self.interpolation_factor = tf.placeholder_with_default(0.0, shape=[], name="interpolation_factor")

        # precompute style features
        self.style_features_pyramid = []
        with tf.name_scope("pre-style-features"), tf.Session() as sess:
            style_image = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='precompute_style')
            style_image_pre = vgg.preprocess(vgg.rgb2bgr(style_image))
            net = vgg.Vgg19()
            net.build(style_image_pre)
            for style_target in self.style_targets:
                style_target = np.expand_dims(style_target, 0)
                style_features = {}
                for layer in STYLE_LAYERS:
                    fv = sess.run(net.net[layer], feed_dict={style_image: style_target})
                    fv = np.reshape(fv, (-1, fv.shape[3]))
                    gram = np.matmul(fv.T, fv) / fv.size
                    style_features[layer] = gram
                self.style_features_pyramid.append(style_features)

        # Content Loss and Style Loss
        content_bgr = vgg.rgb2bgr(self.content_input)
        content_pre = vgg.preprocess(content_bgr)
        content_net = vgg.Vgg19()
        content_net.build(content_pre)
        content_fv = content_net.net[CONTENT_LAYERS]

        self.preds = netdef.shortcut_interpolation(self.content_input / 255., self.shortcut, self.interpolation_factor)
        preds_bgr = vgg.rgb2bgr(self.preds)
        preds_pre = vgg.preprocess(preds_bgr)
        net = vgg.Vgg19()
        net.build(preds_pre)
        preds_content_fv = net.net[CONTENT_LAYERS]

        self.content_loss = args.content_weight * (2 * tf.nn.l2_loss(
            preds_content_fv - content_fv) / (tf.to_float(tf.size(content_fv)) * args.batch_size)
        )

        self.style_losses = []
        for style_layer in STYLE_LAYERS:
            fv = net.net[style_layer]
            bs, height, width, filters = tf.shape(fv)[0], tf.shape(fv)[1], tf.shape(fv)[2], tf.shape(fv)[3]
            size = height * width * filters
            feats = tf.reshape(fv, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_T, feats) / tf.to_float(size)

            style_gram = tf.to_float(tf.cond(self.shortcut[0],
                lambda: self.style_features_pyramid[2][style_layer],
                lambda: tf.cond(self.shortcut[1],
                    lambda: self.style_features_pyramid[1][style_layer],
                    lambda: self.style_features_pyramid[0][style_layer]
                )
            ))
            self.style_losses.append(args.style_weight * (2 * tf.nn.l2_loss(grams - style_gram) / tf.to_float(tf.size(style_gram))) / args.batch_size)
        self.style_loss = functools.reduce(tf.add, self.style_losses)

        # Total Variational Loss
        tv_y_size = tf.to_float(tf.size(self.preds[:, 1:, :, :]))
        tv_x_size = tf.to_float(tf.size(self.preds[:, :, 1:, :]))
        y_tv = tf.nn.l2_loss(self.preds[:, 1:, :, :] - self.preds[:, :-1, :, :])
        x_tv = tf.nn.l2_loss(self.preds[:, :, 1:, :] - self.preds[:, :, :-1, :])
        self.tv_loss = 2 * args.tv_weight * (x_tv / tv_x_size + y_tv / tv_y_size) / args.batch_size
        self.loss = tf.add_n([self.content_loss, self.style_loss, self.tv_loss], name="loss")

    def train(self, args):
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

        for iter_count in xrange(1, int(args.max_iter) + 1):
            feed_dict = self.data_loader.fill_feed_dict(
                self.content_input,
                img_size=DEFAULT_RESOLUTIONS[1] + (3,)
            )
            feed_dict[self.shortcut] = [iter_count % 3 == 2, iter_count % 3 == 1]
            feed_dict[self.interpolation_factor] = 0.0

            _, content_loss, tv_loss, total_loss, style_losses_list = self.sess.run([
                self.optimizer,
                self.content_loss,
                self.tv_loss,
                self.loss,
                self.style_losses
            ], feed_dict=feed_dict)

            if iter_count % args.iter_print == 0 and iter_count != 0:
                print ('Iteration {} / {}\n\tContent loss: {}'.format(iter_count, args.max_iter, content_loss))
                for idx, sloss in enumerate(style_losses_list):
                    print ('\tStyle {} loss: {}'.format(idx, sloss))
                print ('\tTV loss: {}'.format(tv_loss))
                print ('\tTotal loss: {}'.format(total_loss))

            if iter_count % args.checkpoint_iterations == 0 and iter_count != 0:
                self.save(args.checkpoint_dir, iter_count)
                self.save_sample_train(args, join(args.serial, "out_{}_768px.jpg".format(iter_count)), shortcut=STROKE_SHORTCUT_DICT["768"])
                self.save_sample_train(args, join(args.serial, "out_{}_512px.jpg".format(iter_count)), shortcut=STROKE_SHORTCUT_DICT["512"])
                self.save_sample_train(args, join(args.serial, "out_{}_256px.jpg".format(iter_count)), shortcut=STROKE_SHORTCUT_DICT["256"])
            # ...

    def finetune_model(self, args):
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        start_step = 0
        if self.load(args.checkpoint_dir):
            print "[*] Success load the checkpoint {}, continue to train.".format(args.checkpoint_dir)
            checkpoint_names = [f for f in os.listdir(args.checkpoint_dir) if ".meta" in f]
            checkpoint_nums = [int(''.join(x for x in r if x.isdigit())) for r in checkpoint_names]
            start_step = max(checkpoint_nums) + 1
        else:
            print "[!] Error in loading checkpoint"
            return

        for iter_count in xrange(start_step, int(args.max_iter) + start_step + 1):
            feed_dict = self.data_loader.fill_feed_dict(
                self.content_input,
                img_size=DEFAULT_RESOLUTIONS[1] + (3,)
            )
            feed_dict[self.shortcut] = [iter_count % 3 == 2, iter_count % 3 == 1]
            feed_dict[self.interpolation_factor] = 0.0

            _, content_loss, tv_loss, total_loss, style_losses_list = self.sess.run([
                self.optimizer,
                self.content_loss,
                self.tv_loss,
                self.loss,
                self.style_losses
            ], feed_dict=feed_dict)

            if iter_count % args.iter_print == 0 and iter_count != 0:
                print ('Iteration {} / {}\n\tContent loss: {}'.format(iter_count, int(args.max_iter) + start_step, content_loss))
                for idx, sloss in enumerate(style_losses_list):
                    print ('\tStyle {} loss: {}'.format(idx, sloss))
                print ('\tTV loss: {}'.format(tv_loss))
                print ('\tTotal loss: {}'.format(total_loss))

            if iter_count % args.checkpoint_iterations == 0 and iter_count != 0:
                self.save(args.checkpoint_dir, iter_count)
                self.save_sample_train(args, join(args.serial, "out_{}_768px.jpg".format(iter_count)), shortcut=STROKE_SHORTCUT_DICT["768"])
                self.save_sample_train(args, join(args.serial, "out_{}_512px.jpg".format(iter_count)), shortcut=STROKE_SHORTCUT_DICT["512"])
                self.save_sample_train(args, join(args.serial, "out_{}_256px.jpg".format(iter_count)), shortcut=STROKE_SHORTCUT_DICT["256"])


    def load(self, checkpoint_dir):
        print (" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            try:
                self.saver.restore(self.sess, checkpoint_dir)
                return True
            except:
                return False

    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model'), global_step=step)

    def save_sample_train(self, args, output_path, shortcut):
        img = np.array(load_image(args.sample_path, 1024), dtype=np.float32)
        border = np.ceil(np.shape(img)[0]/20/4).astype(int) * 5
        #container = np.ones((args.batch_size, np.shape(img)[0] + 2 * border, np.shape(img)[1] + 2 * border, 3), dtype=np.float32)
        container = [imresize(img, (np.shape(img)[0] + 2 * border, np.shape(img)[1] + 2 * border, 3))]
        container[0][border : np.shape(img)[0] + border, border : np.shape(img)[1] + border, :] = img
        container = np.repeat(container, args.batch_size, 0)

        preds = self.sess.run(self.preds, feed_dict={self.content_input: container, self.shortcut: shortcut, self.interpolation_factor: 0.0})
        
        save_image(output_path, np.squeeze(preds[0][border : np.shape(img)[0] + border, border : np.shape(img)[1] + border, :]))
        print ("[*] Save to {}".format(output_path))
