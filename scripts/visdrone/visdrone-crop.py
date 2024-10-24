import numpy as np
import cv2
import os
from tqdm import tqdm


def check(rgb_path, ir_path):
    if not os.path.exists(rgb_path):
        raise ValueError
    if not os.path.exists(ir_path):
        raise ValueError


def mkdir(rgb_path, ir_path):
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    if not os.path.exists(ir_path):
        os.makedirs(ir_path)


def crop(in_image, out_image):
    image = cv2.imread(in_image)
    cropped = image[100:612, 100:740]
    cv2.imwrite(out_image, cropped)


# ########## train ##########
rgb_input = "../train/trainimg"
rgb_output = "cropped/train/trainimg"
ir_input = "../train/trainimgr"
ir_output = "cropped/train/trainimgr"
# ########## val ##########
# rgb_input = "../val/valimg"
# rgb_output = "cropped/val/valimg"
# ir_input = "../val/valimgr"
# ir_output = "cropped/val/valimgr"
# ########## test ##########
# rgb_input = "../test/testimg"
# rgb_output = "cropped/test/testimg"
# ir_input = "../test/testimgr"
# ir_output = "cropped/test/testimgr"


check(rgb_input, ir_input)
mkdir(rgb_output, ir_output)

rgb_images = [
    (os.path.join(rgb_input, x), os.path.join(rgb_output, x))
    for x in os.listdir(rgb_input)
]

ir_images = [
    (os.path.join(ir_input, x), os.path.join(ir_output, x))
    for x in os.listdir(ir_input)
]
# 转化所有图片
print("Converting visible images...")
for path in tqdm(rgb_images):
    crop(path[0], path[1])


print("Converting infrared images...")
for path in tqdm(ir_images):
    crop(path[0], path[1])
