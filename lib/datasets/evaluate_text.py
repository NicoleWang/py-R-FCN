import sys
import os
import numpy as np
import string
import cPickle
from utils.cython_bbox import bbox_overlaps

def trans_txt_to_pkl(indir, namelist, outpath):
    with open(namelist, 'r') as f:
        names = [x.strip() for x in f.readlines()]
    all_boxes = []
    for name in names:
        txtpath = os.path.join(indir, name)
        with open(txtpath, 'r') as f:
            lines = [[string.atoi(y) for y in x.strip().split()]
                    for x in f.readlines()]
        boxes = np.array(lines)
        all_boxes.append(boxes)
    with open(outpath, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

def evaluate_pkl(gtpath, respath):
    if os.path.splitext(gtpath)[1] != '.pkl':
        print "gt must be in cPickle format\n call trans_txt_to_pkl first"
        exit()
    if os.path.splitext(respath)[1] != '.pkl':
        print "results must be in cPickle format, please check lib/fast_rcnn/test.py"
        exit()

    with open(gtpath, 'rb') as f:
        gt_boxes = cPickle.load(f)
    #gt_boxes[:, 2:3] = gt_boxes[:, 0:1] + gt_boxes[:, 2:3] - 1 ##### WRONG #######
    with open(respath, 'rb') as f:
        res_boxes = cPickle.load(f)

    gt_char_num = 0; #original gt char num
    res_char_num = 0; #result char num
    valid_gt_char_num = 0;
    valid_res_char_num = 0; # after score thresh, remaining char num
    #print gt_boxes[1]
    for i in xrange(0, len(gt_boxes)):
        print "processing %dth test sample" % i
        gt = gt_boxes[i]
        gt[:, 2] = gt[:, 0] + gt[:, 2] - 1
        gt[:, 3] = gt[:, 1] + gt[:, 3] - 1
        res = res_boxes[i]
        gt_char_num += gt.shape[0]
        res_char_num += res.shape[0]
        overlaps = bbox_overlaps(np.ascontiguousarray(res, dtype=np.float),
                             np.ascontiguousarray(gt, dtype=np.float))

        gt_argmax_overlap = overlaps.argmax(axis=0)
        gt_overlap = overlaps[gt_argmax_overlap, np.arange(gt.shape[0])]
        valid_gt = np.where(gt_overlap >= area_thresh)
        valid_gt_char_num += valid_gt[0].shape[0]

        res_argmax_overlap = overlaps.argmax(axis=1)
        res_overlap = overlaps[np.arange(res.shape[0]), res_argmax_overlap]
        valid_res = np.where(res_overlap >= area_thresh)
        valid_res_char_num += valid_res[0].shape[0]

    recall = valid_gt_char_num * 1.0 / gt_char_num
    precision = valid_res_char_num * 1.0 / res_char_num
    print "recall: %f \n precision: %f" %(recall, precision)

def evaluate_dir(imgdir, gtpath, respath, area_thresh, outfile):
    gtfiles = os.listdir(gtpath)
    all_gt_num = 0;
    all_res_num = 0;
    all_recall_num = 0;
    all_precise_num = 0;
    out_fn = open(outfile, 'w')
    for gtfile in gtfiles:
        img_gt_num = 0.000001;
        img_res_num = 0.000001;
        img_recall_num = 0;
        img_precise_num = 0;

        ############### Load ground truth files ################
        gtfilepath = os.path.join(gtpath, gtfile)
        gtlist = []
        with open(gtfilepath) as f:
            lines = [x.strip().split() for x in f.readlines()]
        for line in lines:
            gtlist.append([string.atoi(line[0]), string.atoi(line[1]), string.atoi(line[2]), string.atoi(line[3])])
        gtdets = np.array(gtlist)
        gtdets[:, 2] = gtdets[:, 0] + gtdets[:, 2] - 1;
        gtdets[:, 3] = gtdets[:, 1] + gtdets[:, 3] - 1;
        img_gt_num = gtdets.shape[0]
        all_gt_num = all_gt_num + img_gt_num
        #print gtdets
        #print gtdets.shape

        ############### Load result files #################
        resfilepath = os.path.join(respath, gtfile)
        print resfilepath
        reslist = []
        with open(resfilepath) as f:
            lines = [x.strip().split() for x in f.readlines()]
        for line in lines:
            reslist.append([string.atoi(line[0]), string.atoi(line[1]), string.atoi(line[2]), string.atoi(line[3])])
        if len(reslist) == 0:
            print "No detections"
            print "***********************"
            print "image: ", gtfile[:-4]
            print "image recall: ", img_recall_num * 1.0 / img_gt_num
            print "image precision: ", img_precise_num * 1.0 / img_res_num
            print "***********************\n"
            out_fn.write("*******************************************************************\n")
            out_fn.write(gtfile[:-4])
            out_fn.write("\n")
            out_fn.write("gt char  num: %d\n"%int(img_gt_num))
            out_fn.write("res char num: %d\n"%int(img_res_num))
            out_fn.write("recall  num: %d\n"%img_recall_num)
            out_fn.write("precise num: %d\n"%img_precise_num)
            out_fn.write("recall: %f\n"%(img_recall_num * 1.0 / img_gt_num))
            out_fn.write("precision: %f\n"%(img_precise_num * 1.0 / img_res_num))
            out_fn.write("*******************************************************************\n\n\n")
            continue
        resdets = np.array(reslist)

        #print resdets
        #resdets[:, 2] = resdets[:, 0] + resdets[:, 2] - 1;
        #resdets[:, 3] = resdets[:, 1] + resdets[:, 3] - 1;
        img_res_num = resdets.shape[0]
        all_res_num = all_res_num + img_res_num;
        #print resdets
        #print resdets.shape

        ############## compute one-to-one overlap ratio between all gt and result chars #############
        overlaps = bbox_overlaps(np.ascontiguousarray(resdets, dtype=np.float),
                             np.ascontiguousarray(gtdets, dtype=np.float))
        ### gt num: N, res num, K, then overlaps dim K*N
        #print overlaps

        ########### compute recall num ###########
        gt_argmax_overlap = overlaps.argmax(axis=0)
        #print gt_argmax_overlap;
        gt_overlap = overlaps[gt_argmax_overlap, np.arange(gtdets.shape[0])]
        #print gt_overlap
        valid_gt = np.where(gt_overlap >= area_thresh)
        img_recall_num = valid_gt[0].shape[0]
        all_recall_num = all_recall_num + img_recall_num;
        #print valid_gt
        #print "img_recall_num: ", img_recall_num

        ########### compute precise num ###########
        res_argmax_overlap = overlaps.argmax(axis=1)
        #print res_argmax_overlap
        res_overlap = overlaps[np.arange(resdets.shape[0]), res_argmax_overlap]
        #print res_overlap
        valid_res = np.where(res_overlap >= area_thresh)
        #print valid_res
        img_precise_num = valid_res[0].shape[0]
        all_precise_num = all_precise_num + img_precise_num
        #print "img_precise_num: ", img_precise_num

        print "***********************"
        print "image: ", gtfile[:-4]
        print "image recall: ", img_recall_num * 1.0 / img_gt_num
        print "image precision: ", img_precise_num * 1.0 / img_res_num
        print "***********************\n"
        out_fn.write("*******************************************************************\n")
        out_fn.write(gtfile[:-4])
        out_fn.write("\n")
        out_fn.write("gt char  num: %d\n"%int(img_gt_num))
        out_fn.write("res char num: %d\n"%int(img_res_num))
        out_fn.write("recall  num: %d\n"%img_recall_num)
        out_fn.write("precise num: %d\n"%img_precise_num)
        out_fn.write("recall: %f\n"%(img_recall_num * 1.0 / img_gt_num))
        out_fn.write("precision: %f\n"%(img_precise_num * 1.0 / img_res_num))
        out_fn.write("*******************************************************************\n\n\n")
    print "total recall: ", all_recall_num * 1.0 / all_gt_num
    print "total precision: ", all_precise_num * 1.0 / all_res_num
    out_fn.write("total recall: %f\n"%(all_recall_num * 1.0 / all_gt_num))
    out_fn.write("total precision: %f\n"%(all_precise_num * 1.0 / all_res_num))
    out_fn.close()


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print "Usage: evaluate_text.py gt.pkl res.pkl outfile mode area_thresh imgdir"
        print "OR"
        print "Usage: evaluate_text.py gtdir  resdir  outfile mode area_thresh imgdir"
        print "mode: PKL, DIR"
        exit()
    gtpath  = sys.argv[1]
    respath = sys.argv[2]
    outfile = sys.argv[3]
    mode = sys.argv[4]
    area_thresh = string.atof(sys.argv[5])
    imgdir = sys.argv[6]
    if mode == "PKL" :
        evaluate_pkl(gtpath, respath);
        gtpath  = sys.argv[1]
        respath = sys.argv[2]
    if mode == "DIR" :
        evaluate_dir(imgdir, gtpath, respath, area_thresh, outfile)
    area_thresh = 0.1
    #print os.path.splitext(gtpath)[1]
    #print os.path.splitext(respath)[1]

