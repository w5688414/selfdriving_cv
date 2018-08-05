import tensorflow as tf

from tensorflow.python_io import TFRecordWriter

import numpy as np
import h5py
import glob
import os
from tqdm import tqdm

from IPython import embed

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

input_roots = '/data/dataTrain/val_*/'
output_name = '/data/dataTrain/val.tfrecords'

writer = TFRecordWriter(output_name)

h5files = glob.glob(os.path.join(input_roots, '*.h5'))

for h5file in tqdm(h5files):
    try:
        data = h5py.File(h5file, 'r')
        for i in range(200):
            img     = data['CameraRGB'][i]
            target  = data['targets'][i]

            feature_dict = {'image': _bytes_feature(img.tostring()),
                            'targets': _float_feature(target)}

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())
        data.close()
    except:
        print('filename: {}'.format(h5file))


writer.close()
