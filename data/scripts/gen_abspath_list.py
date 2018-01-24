import sys,os,json
imgdir = sys.argv[1]
txtdir = sys.argv[2]
out_imgfile = sys.argv[3]
out_txtfile = sys.argv[4]

namelist = os.listdir(txtdir)
total = len(namelist)
img_fn = open(out_imgfile, 'w')
txt_fn = open(out_txtfile, 'w')
for idx, name in enumerate(namelist):
    '''
    if idx > 5:
        break
    '''
    print "process %d/%d"%(idx, total)
    prefix = name[:-4]
    imname = prefix+"jpg"
    txt_path = os.path.join(txtdir, name)
    img_path = os.path.join(imgdir, imname)
    txt_abspath = os.path.abspath(txt_path)
    img_abspath = os.path.abspath(img_path)
    print img_abspath, txt_abspath
    img_fn.write("%s\n"%img_abspath)
    txt_fn.write("%s\n"%txt_abspath)
