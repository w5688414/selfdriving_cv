import numpy as np
import tensorflow as tf

from network import make_network
from data_provider import DataProvider
from tensorflow.core.protobuf import saver_pb2

import time
import os

log_path = './log'
save_path = './data'

if __name__ == '__main__':

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        train_provider = DataProvider('/data/dataTrain/train.tfrecords', sess)
        val_provider = DataProvider('/data/dataTrain/val.tfrecords', sess)

        network = make_network()

        lr = 0.0001
        lr_placeholder = tf.placeholder(tf.float32, [])
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_placeholder,
                beta1=0.7, beta2=0.85)
        optimizer = optimizer.minimize(network['loss'])

        sess.run(tf.global_variables_initializer())
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
        saver.restore(sess, os.path.join(save_path, 'step-7500.ckpt'))

        step = 0

        while True:
            if step % 50 == 0:
                val_batch = val_provider.get_minibatch()
                val_loss = sess.run(network['loss'],
                        feed_dict={network['inputs'][0]: val_batch.images,
                                   network['inputs'][1]: val_batch.data[0],
                                   network['labels'][0]: val_batch.data[1]})
                print('VALIDATION--------loss: %.4f' %  val_loss)
            if step % 500 == 0:
                model_path = os.path.join(save_path, 'step-%d.ckpt' % step)
                saver.save(sess, model_path)
                print("Checkpoint saved to %s" % model_path)

            a = time.time()
            batch = train_provider.get_minibatch(augment=True)
            imgs = batch.images
            speed, target_control, _ = batch.data
            b = time.time()
            _, train_loss = sess.run([optimizer, network['loss']],
                    feed_dict={network['inputs'][0]: imgs,
                               network['inputs'][1]: speed,
                               network['labels'][0]: target_control,
                               lr_placeholder: lr})
            c = time.time()
            print('step: %d loss %.4f prepare: %.3fs gpu: %.3fs' % (step, train_loss, b-a, c-b))

            step += 1


