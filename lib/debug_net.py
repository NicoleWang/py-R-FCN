from ctypes import *
cdll.LoadLibrary('/temp-hdd/tangtang/cuda_deploy/cuda-8.0/lib64/libcudart.so.8.0')
cdll.LoadLibrary('/temp-hdd/tangtang/cuda_deploy/cuda-8.0/lib64/libcudnn.so.5')
import sys
sys.path.insert(0, '/home/wangyuzhuo/Experiments/py-R-FCN/lib/track')
import track.track_data_layer
import numpy as np
import caffe
caffe.set_mode_cpu()
net = caffe.Net('./track/models/siamese_test.prototxt', '../models/siamese__siamese_iter_500.caffemodel', caffe.TEST)
img = np.random.rand(1,3,127,127)
net.forward(data=img)
print net.blobs['conv5'].data
