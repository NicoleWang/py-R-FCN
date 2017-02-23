import sys
import os
import numpy as np
import string
import cPickle
from utils.cython_bbox import bbox_overlaps
from utils.cython_bbox import bbox_intersections

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

def evaluate_dir(imgdir, gtpath, respath, area_thresh, outfile):
    gtfiles = os.listdir(gtpath)
    all_recall_ratio = 0;
    all_precise_ratio = 0;
    out_fn = open(outfile, 'w')
    for gtfile in gtfiles:
        img_gt_area = 0.000001;
        img_res_area = 0.000001;
        img_recall_area = 0;
        img_precise_area = 0;

        ############### Load ground truth files ################
        gtfilepath = os.path.join(gtpath, gtfile)
        gtlist = []
        with open(gtfilepath) as f:
            lines = [x.strip().split() for x in f.readlines()]
        for line in lines:
            gtlist.append([string.atoi(line[0]), string.atoi(line[1]), string.atoi(line[2]), string.atoi(line[3])])
        gtdets = np.array(gtlist)
        gtareas = gtdets[:, 2] * gtdets[:, 3]
        print gtareas
        img_gt_area = np.sum(gtareas)

        gtdets[:, 2] = gtdets[:, 0] + gtdets[:, 2] - 1;
        gtdets[:, 3] = gtdets[:, 1] + gtdets[:, 3] - 1;

        ############### Load result files #################
        resfilepath = os.path.join(respath, gtfile)
        print resfilepath
        reslist = []
        with open(resfilepath) as f:
            lines = [x.strip().split() for x in f.readlines()]
        for line in lines:
            reslist.append([string.atoi(line[0]), string.atoi(line[1]), string.atoi(line[2]), string.atoi(line[3])])
        if len(reslist) == 0:
            continue;
        resdets = np.array(reslist)
        print resdets
        resareas = resdets[:,2] * resdets[:,3]
        img_res_area = np.sum(resareas)

        #resdets[:, 2] = resdets[:, 0] + resdets[:, 2] - 1;
        #resdets[:, 3] = resdets[:, 1] + resdets[:, 3] - 1;

        ############## compute one-to-one overlap ratio between all gt and result chars #############
        overlaps = bbox_intersections(np.ascontiguousarray(resdets, dtype=np.float),
                             np.ascontiguousarray(gtdets, dtype=np.float))
        ### gt num: N, res num, K, then overlaps dim K*N
        print overlaps

        ########### compute recall num ###########
        gt_argmax_overlap = overlaps.argmax(axis=0)
        #print gt_argmax_overlap;
        gt_overlap = overlaps[gt_argmax_overlap, np.arange(gtdets.shape[0])]
        print gt_overlap
        valid_gt = np.where(gt_overlap >= area_thresh)
        img_recall_area = np.sum(gt_overlap)
        #print valid_gt
        #print "img_recall_num: ", img_recall_num

        ########### compute precise num ###########
        res_argmax_overlap = overlaps.argmax(axis=1)
        #print res_argmax_overlap
        res_overlap = overlaps[np.arange(resdets.shape[0]), res_argmax_overlap]
        print res_overlap
        valid_res = np.where(res_overlap >= area_thresh)
        #print valid_res
        img_precise_area = np.sum(res_overlap)
        #print "img_precise_num: ", img_precise_num
        img_recall = img_recall_area * 1.0 / img_gt_area
        img_precision = img_precise_area * 1.0 / img_res_area

        all_recall_ratio += img_recall;
        all_precise_ratio += img_precision;
        print "***********************"
        print "image: ", gtfile[:-4]
        print "image recall: ", img_recall
        print "image precision: ", img_precision
        print "***********************\n"
        out_fn.write("*******************************************************************\n")
        out_fn.write(gtfile[:-4])
        out_fn.write("\n")
        out_fn.write("gt area: %d\n"%int(img_gt_area))
        out_fn.write("res area: %d\n"%int(img_res_area))
        out_fn.write("recall  area: %d\n"%img_recall_area)
        out_fn.write("precise area: %d\n"%img_precise_area)
        out_fn.write("recall: %f\n"%(img_recall))
        out_fn.write("precision: %f\n"%(img_precision))
        out_fn.write("*******************************************************************\n\n\n")
    print "total recall: ", all_recall_ratio * 1.0 / len(gtfiles)
    print "total precision: ", all_precise_ratio * 1.0 / len(gtfiles)
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

