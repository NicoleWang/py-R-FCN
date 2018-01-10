import sys, os
imgdir = sys.argv[1]
namelist = os.listdir(imgdir)
num_total = len(namelist)
fn = open("view.html", 'w')
fn.write("<head> <meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" /> </head>")
fn.write("\n")
for idx, name in enumerate(namelist):
    print "%d/%d" %(idx, num_total)
    imgpath = os.path.join(imgdir, name)
    img_url = "<img src=\"%s\" height=\"200\" width=\"200\"></img>"%imgpath
    fn.write("%s\n"%img_url)
fn.close
