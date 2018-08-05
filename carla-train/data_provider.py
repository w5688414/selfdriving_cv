import tensorflow as tf
import numpy as np
import glob
import os
import h5py
from imgaug.imgaug import Batch, BatchLoader, BackgroundAugmenter
import imgaug.augmenters as iaa
import cv2

from IPython import embed

BATCHSIZE = 120

st = lambda aug: iaa.Sometimes(0.4, aug)
oc = lambda aug: iaa.Sometimes(0.3, aug)
rl = lambda aug: iaa.Sometimes(0.09, aug)

seq = iaa.Sequential([
        rl(iaa.GaussianBlur((0, 1.5))),
        rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),
        oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),
        oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)),
        oc(iaa.Add((-40, 40), per_channel=0.5)),
        st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),
        rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
], random_order=True)

'''
def augmentation(imgs):
    return imgs
'''

def parse_proto(example_proto):
    features = tf.parse_single_example(example_proto,
            features={'image':      tf.FixedLenFeature([], tf.string),
                      'targets':    tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)})
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [88, 200, 3])

    speed           = features['targets'][10]
    target_control  = features['targets'][0:3]
    target_command  = features['targets'][24] % 4
    return image, speed[None], target_control, target_command

class DataProvider:
    def __init__(self, filename, session):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.repeat().shuffle(buffer_size=2000).map(parse_proto).batch(BATCHSIZE)
        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)
        dataset_init = iterator.make_initializer(dataset)
        session.run(dataset_init)

        self.dataset    = dataset
        self.session    = session
        self.next       = iterator.get_next()

    def get_minibatch(self, augment = False):
        data = self.session.run(self.next)
        imgs = data[0].astype('float32')
        if augment:
            imgs = seq.augment_images(imgs)
        return Batch(images=imgs, data=data[1:])

    def show_imgs(self):
        batch = self.get_minibatch(True)
        for img in batch.images:
            cv2.imshow('img', img)
            cv2.waitKey(0)

# Test tf.data & imgaug backgroud loader APIs
if __name__ == '__main__':
    import time
    sess = tf.Session()
    dp = DataProvider('/mnt/AgentHuman/train.tfrecords', sess)

    while True:
        a = time.time()
        dp.get_minibatch()
        b = time.time()
        print(b-a)

