# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""
import sys, os, string
sys.path.insert(0, '/home/wangyuzhuo/ENVS/caffe-for-rfcn/python/')
sys.path.append('/home/wangyuzhuo/projects/Big_Human_Detection/lib/')
#import _init_paths
import numpy as np
import cv2
import caffe
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from utils.blob import im_list_to_blob

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors

def im_detect(net, im):
    blobs, im_scales = _get_blobs(im)
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)
    rois = net.blobs['rois'].data.copy()
    boxes = rois[:, 1:5] / im_scales[0]
    scores = blobs_out['cls_prob'] ###rfcnn
    box_deltas = blobs_out['bbox_pred'] #rfcnn
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes

def check_rois(net, im):
    blobs, im_scales = _get_blobs(im)
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)
    rois1 = net.blobs['rois1'].data.copy()
    rois2 = net.blobs['rois2'].data.copy()
    rois3 = net.blobs['rois3'].data.copy()
    roi_scores1 = net.blobs['roi_scores1'].data.copy()
    roi_scores2 = net.blobs['roi_scores2'].data.copy()
    roi_scores3 = net.blobs['roi_scores3'].data.copy()
    #rois = np.vstack((rois1, rois2, rois3))
    #roi_scores = np.vstack((roi_scores1, roi_scores2, roi_scores3))
    rois = rois2
    roi_scores = roi_scores2
    boxes = rois[:, 1:5] / im_scales[0]
    #print boxes.shape
    #print rois_scores.shape
    #scores = blobs_out['cls_prob'] ###rfcnn
    #boxes = clip_boxes(boxes, im.shape)
    return roi_scores, boxes

def vis_detections(im, dets, save_path, thresh=0.3):
    """Visual debugging of detections."""
    for i in xrange(0, dets.shape[0]):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    cv2.imwrite(save_path, im)

def post_bbox(img, scores, bboxes, outpath):
    thresh = 0.5
    inds = np.where(scores[:, 1] > thresh)[0]
    cls_scores = scores[inds, 1]
    cls_boxes = boxes[inds, 4:8]
    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
    keep = nms(cls_dets, cfg.TEST.NMS)
    cls_dets = cls_dets[keep, :]
    prefix = outpath[0:-4]
    out_txt = prefix+".txt"
    write_result(cls_dets, out_txt)
    vis_detections(img,  cls_dets, outpath, 0.5)

def post_rois(img, scores, bboxes, outpath):
    thresh = 0.5
    inds = np.where(scores[:, 0] > thresh)[0]
    cls_scores = scores[inds, 0]
    cls_boxes = boxes[inds, :]
    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
    keep = nms(cls_dets, cfg.TEST.NMS)
    cls_dets = cls_dets[keep, :]
    print cls_dets
    prefix = outpath[0:-4]
    vis_detections(img,  cls_dets, outpath, 0.5)
    #out_txt = prefix+".txt"
    #write_result(cls_dets, out_txt)

def write_result(boxes, save_path):
    f = open(save_path, 'w')
    if boxes.shape[0] > 1:
        print boxes
    box = boxes[0, 0:4].astype(int)
    score = boxes[0, 4]
    f.write("%d\t%d\t%d\t%d\t%f\n" % (box[0], box[1], box[2], box[3], score))
    '''
    for i in xrange(0, boxes.shape[0]):
        box = boxes[i, 0:4].astype(int)
        score = boxes[i, 4]
        if score >= 0.5:
            f.write("%d\t%d\t%d\t%d\t%f\n" % (box[0], box[1], box[2], box[3], score))
    '''
    f.close()

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "python test_human.py proto model gpu imgdir outdir"
        exit()

    ##input arguments
    deploy_proto = sys.argv[1]
    human_model = sys.argv[2]
    gpu_id = string.atoi(sys.argv[3])
    imgdir = sys.argv[4]
    outdir = sys.argv[5]

    # set device  mode
    if gpu_id >=0 and gpu_id <= 3:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        caffe.set_mode_cpu()

    #init detection model
    net = caffe.Net(deploy_proto, human_model, caffe.TEST)

    namelist = os.listdir(imgdir)
    for name in namelist:
        print "processing ", name
        imgpath = os.path.join(imgdir, name)
        img = cv2.imread(imgpath)
        #scores, boxes = im_detect(net, img)
        scores, boxes = check_rois(net, img)
        save_path = os.path.join(outdir, name)
        #post_bbox(img, scores, boxes, save_path)
        post_rois(img, scores, boxes, save_path)

