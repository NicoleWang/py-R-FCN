# --------------------------------------------------------
# Data layer for object tracking
# Written by Wang Yuzhuo
# --------------------------------------------------------

"""The data layer used during training to train an object tracking network.

TrackDataLayer implements a Caffe Python layer.
"""

import caffe
#from fast_rcnn.config import cfg
import track.track_cfg as tcfg
#from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
import os, string
import cv2
#from multiprocessing import Process, Queue

class TrackDataLayer(caffe.Layer):
    """Tracking data layer used for training."""
    def _gen_label(self):
        stride = 8
        pos_thresh = tcfg.POS_THRESH / stride
        ctr = np.ceil(self._label_size / 2)
        cx = ctr
        cy = ctr
        label = np.zeros([self._label_size, self._label_size])
        instance_weight = np.ones([self._label_size, self._label_size])
        for i in xrange(0,self._label_size):
            for j in xrange(0, self._label_size):
                dist = np.sqrt((i - cx)*(i - cx) +  (j - cy)*(j - cy))
                if dist <= pos_thresh:
                    label[i, j] = 1
                else:
                    label[i, j] = -1
        pos_idx = np.where(label == 1)
        pos_num = pos_idx[0].shape[0]
        total_num = self._label_size * self._label_size;
        neg_num = total_num - pos_num
        for i in xrange(0, self._label_size):
            for j in xrange(0, self._label_size):
                if label[i, j] == 1:
                    instance_weight[i, j] = 0.5 / pos_num
                else:
                    instance_weight[i, j] = 0.5 / neg_num
        #label = label / pos_num
        return label, instance_weight

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        #if self._cur_idx >= 2:
        #    exit()
        img_list = self._img_list[self._cur_idx:(self._cur_idx + self._batchsize)]
        #img_list = self._img_list[0:1]
        #print(img_list)
        z_blob = np.zeros([self._batchsize, 3, self._exemplar_size, self._exemplar_size])
        x_blob = np.zeros([self._batchsize, 3, self._search_size, self._search_size])
        label_blob = np.zeros([self._batchsize, 1, self._label_size, self._label_size])
        label, instance_weight = self._gen_label()
        for ind, x_im_path in enumerate(img_list):
            z_im_path = x_im_path.replace("x", "z")
            #print x_im_path
            #print z_im_path
            #mean = np.array([[[102.9801, 115.9465, 122.7717]]]);
            x_im = cv2.imread(x_im_path)
            z_im = cv2.imread(z_im_path)
            x_im = cv2.resize(x_im, (255,255))
            z_im = cv2.resize(z_im, (127,127))
            #x_im = x_im - mean;
            #z_im = z_im - mean;
            xshape = x_im.shape
            zshape = z_im.shape
            #print "before reshape", x_im.shape
            x_im = np.reshape(x_im, [1, xshape[0], xshape[1], xshape[2]])
            #print "after reshape", x_im.shape
            #x_im[np.newaxis, :, :, :]
            x_im = np.swapaxes(x_im, 0, 3)
            #print "after swap", x_im.shape
            x_im = np.reshape(x_im, [xshape[2], xshape[0], xshape[1]])
            z_im = np.reshape(z_im, [1, zshape[0], zshape[1], zshape[2]])
            #z_im[np.newaxis, :, :, :]
            z_im = np.swapaxes(z_im, 0, 3)
            #print "after", z_im.shape
            z_im = np.reshape(z_im, [zshape[2], zshape[0], zshape[1]])
            #print "x_im: "
            #print x_im[0,:,:]
            #print "z_im: "
            #print z_im[0,:,:]
            z_blob[ind,:,:,:] = z_im[:,:,:]
            x_blob[ind,:,:,:] = x_im[:,:,:]
            label_blob[ind,0, :, :] = label
            label_blob = np.reshape(label_blob, [self._label_size*self._label_size, 1,1,1])
        blobs = {'exemplar':z_blob, 'search':x_blob, 'label':label_blob, 'instance':instance_weight}
        self._cur_idx += self._batchsize
        return blobs

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def create_image_list(self):
        self._img_list_path = tcfg.IMG_LIST_PATH
        self._train_dir = tcfg.TRAIN_DIR
        with open(self._img_list_path, 'r') as fn:
            self._img_list = [os.path.join(self._train_dir, x.strip()) for x in fn.readlines()]
            self._img_list.sort()
            import random
            random.shuffle(self._img_list)
            #self._img_list = [self._img_list[0]]
            #for i in xrange(0,222):
            #    print self._img_list[i]

    def setup(self, bottom, top):
        """Setup the TrackDataLayer."""
        self.create_image_list()

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        print layer_params
        #self._label_size = string.atoi(layer_params['label_size'])
        self._label_size = 17
        self._exemplar_size = tcfg.Z_SIZE
        self._search_size = tcfg.X_SIZE
        self._batchsize = tcfg.BATCH_SIZE
        self._name_to_top_map = {}
        self._cur_idx = 0

        # data blob: holds a batch of N images, each with 3 channels
        #top[0]: hold exemplar images
        idx = 0
        top[idx].reshape(self._batchsize, 3,
            self._exemplar_size, self._exemplar_size)
        self._name_to_top_map['exemplar'] = idx
        idx += 1

        top[idx].reshape(self._batchsize, 3,
            self._search_size, self._search_size)
        self._name_to_top_map['search'] = idx
        idx += 1
        #top[idx].reshape(self._batchsize, 1, self._label_size, self._label_size)
        top[idx].reshape(self._label_size * self._label_size, 1, 1, 1)
        self._name_to_top_map['label'] = idx
        idx += 1
        top[idx].reshape(self._label_size * self._label_size, 1, 1, 1)
        self._name_to_top_map['instance'] = idx
        idx += 1

        print 'TrackDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            shape = blob.shape
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
            #if blob_name == 'label':
            #    print blob
            #print "image: "
            #print blob
            #print blob_name
            #print top[top_ind].data[0,0,:,:]

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
