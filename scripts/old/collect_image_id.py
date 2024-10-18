# -*- coding: utf-8 -*-

import os
import argparse

base_img_path = str("/hy-tmp/visdrone/")
parser = argparse.ArgumentParser()
parser.add_argument('--set_name', default="", type=str, help='image set name')
opt = parser.parse_args()
img_set_name = opt.set_name
img_set_path = base_img_path + img_set_name

if not os.path.exists(img_set_path):
    raise NotADirectoryError

img_train_path = img_set_path + "/train"
img_test_path = img_set_path + "/test"

total_train_img = os.listdir(img_train_path)
total_test_img = os.listdir(img_test_path)

train_index = range(len(total_train_img))
test_index = range(len(total_test_img))


def write_image_id(file_handler, index, img):
    for ind in index:
        name = img[ind][:-4] + '\n'
        file_handler.write(name)


file_train = open(img_set_path + '/train_id.txt', 'w')
file_test = open(img_set_path + '/test_id.txt', 'w')

write_image_id(file_train, train_index, total_train_img)
write_image_id(file_test, test_index, total_test_img)

file_train.close()
file_test.close()

