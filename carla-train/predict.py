import tensorflow as tf
import glob
import h5py
import numpy as np
from network import make_network
# read an example h5 file
datasetDirTrain = '/home/eric/self-driving/AgentHuman/SeqTrain/'
datasetDirVal = '/home/eric/self-driving/AgentHuman/SeqVal/'
datasetFilesTrain = glob.glob(datasetDirTrain+'*.h5')
datasetFilesVal = glob.glob(datasetDirVal+'*.h5')
print("Len train:{0},len val{1}".format(len(datasetFilesTrain),len(datasetFilesVal)))
data = h5py.File(datasetFilesTrain[1], 'r')
image_input = data['rgb'][1]
input_speed =np.array([[100]])
image_input = image_input.reshape(
            (1, 88, 200, 3))

with tf.Session() as sess:
    network = make_network()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint("./data")
    if ckpt:
        saver.restore(sess, ckpt)
    output=sess.run(network['outputs'], feed_dict={network['inputs'][0]:image_input,
    network['inputs'][1]: input_speed})
    print(output)
    sess.close()