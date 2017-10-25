#!/usr/bin/env python3

import tensorflow as tf
from glob import glob
import os.path

class Classifier:
    def __init__(self, name):
        self.name = name

    def exists():
        return tf.train.latest_checkpoint(os.path.join(os.path.expanduser('~/models/cls'), self.name)) is not None

    def build(self):
        self.filename = tf.train.latest_checkpoint(os.path.join(os.path.expanduser('~/models/cls'), self.name))
        self.saver = tf.train.import_meta_graph(self.filename + '.meta', True, 'extcls')
        clscol = tf.get_collection('cls')
        self.image, self.logits, self.loss, self.pred, self.acc = clscol

    def restore(self, sess):
        self.saver.restore(sess, self.filename)
        self.sess = sess

    def get_logits(self, batch):
        logits = self.sess.run([self.logits], feed_dict={self.image: batch})
        return logits

    def get_pred(self, batch):
        pred = self.sess.run([self.pred], feed_dict={self.image: batch})
        return pred


if __name__ == '__main__':
    cls = Classifier('mnist')
    cls.build()
    with tf.Session() as sess:
        cls.restore(sess)
