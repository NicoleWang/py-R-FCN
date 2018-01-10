import sys, os

list_file = sys.argv[1]
indir = sys.argv[2]
outdir = sys.argv[3]

with open(list_file, 'r') as f:
    name_list = [x.strip() for x in f.readlines()]

for idx, name in enumerate(name_list):
    if idx >= 100:
        break
    inpath = os.path.join(indir, name)
    outpath = os.path.join(outdir, name)
    cmd = "cp %s %s"%(inpath, outpath)
    os.system(cmd)
