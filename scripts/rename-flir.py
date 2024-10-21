import os
import re

# filepath = "/hy-tmp/align/AnnotationsYolo"
filepath = "/hy-tmp/align/rgb/images"
# filepath = "/hy-tmp/align/ir/images"

print("FILEPATH: %s" % filepath)
print("##### START RENAME FILES ######")

for label in os.listdir(filepath):
    filename, ext = os.path.splitext(label)
    m = re.match(r"FLIR_(\d+)_.*", filename)
    if m:
        index = m.groups()[0]
        name = "FLIR_" + str(index) + ext
        print("rename {} --> {}".format(label, name))
        os.rename(os.path.join(filepath, label), os.path.join(filepath, name))

