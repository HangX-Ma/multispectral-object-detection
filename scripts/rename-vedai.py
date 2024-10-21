import shutil
import os
import re

filepath = "/hy-tmp/Vehicules512"
rgbpath = "/hy-tmp/VEDAI/visible/images"
irpath = "/hy-tmp/VEDAI/infrared/images"

rgbcount = 0
ircount = 0

if not filepath or not os.path.exists(filepath):
    raise NotADirectoryError

if not os.path.exists(rgbpath):
    os.makedirs(rgbpath)

if not os.path.exists(irpath):
    os.makedirs(irpath)

print("FILEPATH: %s" % filepath)
print("##### START RENAME FILES ######")

for imagefile in os.listdir(filepath):
    # filter out abnormal images
    m = re.match(r"(\d+)_(ir|co)\.png$", imagefile)
    if m:
        index, imgtype = m.groups()
        name = str(index) + ".png"

        src = os.path.join(filepath, imagefile)
        if imgtype == "ir":
            rgbcount += 1
            dst = os.path.join(irpath, name)
        elif imgtype == "co":
            ircount += 1
            dst = os.path.join(rgbpath, name)
        else:
            raise ValueError

        print("[{}]: rename {} --> {}".format(imgtype, src, dst))
        shutil.copy(src, dst)

print("##### DONE [RGB]: {}, [IR]: {} #####".format(rgbcount, ircount))
