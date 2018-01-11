# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
#import utils.cython_bbox
import cPickle
from fast_rcnn.config import cfg
import json

class text(imdb):
    def __init__(self,devkit_path=None):
        imdb.__init__(self, 'human')
        #self._split = split # split : train / test
        #self._image_set = image_set # image_set: chn / eng
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._train_image_list = self._load_name_list(cfg.TRAIN_IMAGE_LIST)
        self._train_label_list = self._load_name_list(cfg.TRAIN_LABEL_LIST)
        #print self._train_image_list
        #print self._train_label_list

        '''
        if self._split == 'train':
            self._data_path = cfg.TRAIN_DIR
        if self._split == 'test':
            self._data_path = cfg.TEST_DIR
        '''

        self._classes = ('__background__', # always index 0
                         'text')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        #self._image_index = self._load_image_set_index()
        self._image_index = self._train_image_list

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        '''
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
        '''

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        #return self.image_path_from_index(self._image_index[i])
        return self._image_index[i]
    def _load_name_list(self, filepath):
        assert os.path.isfile(filepath), \
                'File does not exist: {}'.format(filepath)
        with open(filepath) as f:
            namelist = [x.strip() for x in f.readlines()]
        return namelist

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        ####wangyuzhuo
        #image_path = os.path.join(self._data_path, 'images256', index)
        image_path = os.path.join(self._data_path, 'images', index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
#        image_set_file = os.path.join(self._data_path, 'ImageSets',
#                                      self._split + '.txt')
        #image_set_file = os.path.join(self._data_path, 'imname_list_sampled.txt')
        image_set_file = os.path.join(self._data_path, 'imname_list_distill.txt')
        #image_set_file = os.path.join(self._data_path, 'imname_list.txt')
        print self._devkit_path
        print self._data_path
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        #return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)
        return cfg.DATA_DIR

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb_temp = [self._load_text_annotation(index)
                    for index in self._train_label_list]
        image_index = []
        gt_roidb = []
        for i in xrange(len(gt_roidb_temp)):
            if len(gt_roidb_temp[i]['boxes']) < 1:
                continue
            gt_roidb.append(gt_roidb_temp[i])
            image_index.append(self.image_index[i])
        self.image_index = image_index
        #gt_roidb = gt_roidb_temp

        #gt_roidb = [x for x in gt_roidb_temp if len(x['boxes']) >= 1]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_text_annotation(self, index):
        """
        Load image and bounding boxes info from txt file for chinese text data
        format: left top width height
        """
        ####wangyuzhuo
        #filename = os.path.join(self._data_path, 'annotations256', index[0:-4] + '.json')
        #filename = os.path.join(self._data_path, 'distill_annotations', index[0:-4] + '.json')
        #filename = os.path.join(self._data_path, 'annotations', index[0:-4] + '.json')
        #print index
        with open(index) as f:
            t_data = json.load(f)
            #all_bboxes = [t_data['bbox']]
            all_bboxes = t_data['bbox']
        if len(all_bboxes) == 0:
            print index
            return {'boxes':[]}

        num_boxes = len(all_bboxes)
        boxes = np.zeros((num_boxes, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_boxes), dtype=np.int32)
        overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
        for idx, roi in enumerate(all_bboxes):
            x1 = max(0, roi[0] + 1) ; y1 = roi[1] + 1; x2 = roi[2] - 1; y2 = roi[3] - 1
            cls = self._class_to_ind['text']
            boxes[idx, :] = [x1, y1, x2, y2]
            gt_classes[idx] = cls
            overlaps[idx, cls] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}


    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
