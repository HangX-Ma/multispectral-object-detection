# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import argparse

from numpy.core.defchararray import isspace

classes = ["person", "bicycle", "car"]  # change to your classes


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


def convert_annotation(filename, xmlpath, txtpath):
    in_file = open(xmlpath + '/%s.xml' % filename, encoding='UTF-8')
    out_file = open(txtpath + '/%s.txt' % filename, 'w')
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
    
    in_file.close()
    out_file.close()


parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', default='Annotations', type=str, help='input xml label path')
parser.add_argument('--txt_path', default='AnnotationsYolo', type=str, help='output txt label path')
opt = parser.parse_args()

xmlfile_path = opt.xml_path
txtfile_path = opt.txt_path

if not xmlfile_path or not os.path.exists(xmlfile_path):
    raise NotADirectoryError

if not os.path.exists(txtfile_path):
    os.makedirs(txtfile_path)

for xmlfile in os.listdir(xmlfile_path):
    filename, ext = os.path.splitext(xmlfile)
    convert_annotation(filename, xmlfile_path, txtfile_path)