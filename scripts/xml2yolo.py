# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import argparse

from numpy.core.defchararray import isspace

sets = ['train', 'test']
classes = ["car", "truck", "bus", "van", "freight car"]  # change to your classes


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(_image_id, _img_set_name):
    in_file = open('/hy-tmp/visdrone/Annotations/%s.xml' % _image_id, encoding='UTF-8')
    out_file = open('/hy-tmp/visdrone/' + _img_set_name + '/' + ('labels/%s.txt' % _image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def convert_visdrone(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = abs((box[0] + box[1]) / 2.0 - 1)
    y = abs((box[2] + box[3]) / 2.0 - 1)
    w = abs(box[1] - box[0])
    h = abs(box[3] - box[2])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation_visdrone(_image_id, _img_set_name):
    in_file = open('/hy-tmp/visdrone/Annotations/%s.xml' % _image_id, encoding='UTF-8')
    out_file = open('/hy-tmp/visdrone/' + _img_set_name + '/' + ('labels/%s.txt' % _image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        '''if obj.find('difficult').text is not None:
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue'''  # 数据集里部分标签里没有difficult
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        # 数据集里有三种形式标点
        if obj.find('polygon') is not None:
            xmlbox = obj.find('polygon')
            xmin = int(min((xmlbox.find('x1').text), (xmlbox.find('x2').text), (xmlbox.find('x3').text),
                           (xmlbox.find('x4').text)))
            xmax = int(max((xmlbox.find('x1').text), (xmlbox.find('x2').text), (xmlbox.find('x3').text),
                           (xmlbox.find('x4').text)))
            ymin = int(min((xmlbox.find('y1').text), (xmlbox.find('y2').text), (xmlbox.find('y3').text),
                           (xmlbox.find('y4').text)))
            ymax = int(max((xmlbox.find('y1').text), (xmlbox.find('y2').text), (xmlbox.find('y3').text),
                           (xmlbox.find('y4').text)))
        if obj.find('bndbox') is not None:
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            ymin = int(xmlbox.find('ymin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymax = int(xmlbox.find('ymax').text)
        if obj.find('point') is not None:
            xmlbox = obj.find('point')
            xmin = int(xmlbox.find('x').text)
            ymin = int(xmlbox.find('y').text)
            xmax = int(xmlbox.find('x').text)
            ymax = int(xmlbox.find('y').text)
        b = (xmin, xmax, ymin, ymax)
        print(b)
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert_visdrone((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


base_img_path = str("/hy-tmp/visdrone/")
parser = argparse.ArgumentParser()
parser.add_argument('--set_name', default='', type=str, help='image set name')
opt = parser.parse_args()

# check input arguments
img_set_name = opt.set_name
img_set_path = base_img_path + img_set_name
if not opt.set_name or not os.path.exists(img_set_path):
    raise NotADirectoryError

for set_type in sets:
    if not os.path.exists(img_set_path + '/labels/'):
        os.makedirs(img_set_path + '/labels/')

    image_ids = open(img_set_path + ('/%s_id.txt' % set_type)).read().strip().split()
    list_file = open(img_set_path + ('/%s.txt' % set_type), 'w')

    for image_id in image_ids:
        list_file.write(img_set_path + ("/images/%s.jpg\n" % image_id))
        convert_annotation_visdrone(image_id, img_set_name)
    list_file.close()
