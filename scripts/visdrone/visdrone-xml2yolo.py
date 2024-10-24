# https://blog.csdn.net/qq_45430086/article/details/140601754
# https://blog.csdn.net/Alan_Dr/article/details/135907288
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from tqdm.contrib import tzip


sets = ["train", "val", "test"]
classes = ["car", "truck", "bus", "van", "feright car", "feright"]


def convert(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = abs((box[0] + box[1]) / 2.0 - 1)
    y = abs((box[2] + box[3]) / 2.0 - 1)
    w = abs(box[1] - box[0])
    h = abs(box[3] - box[2])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def xml2yolo(xml, xml_path, txt_path):
    filename, ext = os.path.splitext(xml)
    xml_file = os.path.join(xml_path, xml)
    txt_file = os.path.join(txt_path, filename + ".txt")

    tree = ET.parse(xml_file)  # 解析xml文件 然后转换为DOTA格式文件
    root = tree.getroot()

    # we need to modify to suit cropped image size
    size = root.find("size")
    w = int(size.find("width").text) - 200
    h = int(size.find("height").text) - 200

    with open(txt_file, "w+", encoding="UTF-8") as out_file:
        for obj in root.findall("object"):
            # deal with 'difficult' tag
            obj_difficult = obj.find("difficult")
            if obj_difficult:
                difficult = obj_difficult.text
            else:
                difficult = "0"

            # NOTE: rename 'feright car' to 'freight_car' and avoid '*' and single 'feright'
            name = obj.find("name").text
            if name not in classes:
                continue
            elif name == "feright":
                name = "feright car"
            else:
                name = name

            cls_id = classes.index(name)

            # 数据集里有三种形式标点
            if obj.find("polygon") is not None:
                xmlbox = obj.find("polygon")
                xmin = int(
                    min(
                        (xmlbox.find("x1").text),
                        (xmlbox.find("x2").text),
                        (xmlbox.find("x3").text),
                        (xmlbox.find("x4").text),
                    )
                )
                xmax = int(
                    max(
                        (xmlbox.find("x1").text),
                        (xmlbox.find("x2").text),
                        (xmlbox.find("x3").text),
                        (xmlbox.find("x4").text),
                    )
                )
                ymin = int(
                    min(
                        (xmlbox.find("y1").text),
                        (xmlbox.find("y2").text),
                        (xmlbox.find("y3").text),
                        (xmlbox.find("y4").text),
                    )
                )
                ymax = int(
                    max(
                        (xmlbox.find("y1").text),
                        (xmlbox.find("y2").text),
                        (xmlbox.find("y3").text),
                        (xmlbox.find("y4").text),
                    )
                )
            if obj.find("bndbox") is not None:
                xmlbox = obj.find("bndbox")
                xmin = int(xmlbox.find("xmin").text)
                ymin = int(xmlbox.find("ymin").text)
                xmax = int(xmlbox.find("xmax").text)
                ymax = int(xmlbox.find("ymax").text)
            if obj.find("point") is not None:
                xmlbox = obj.find("point")
                xmin = int(xmlbox.find("x").text)
                ymin = int(xmlbox.find("y").text)
                xmax = int(xmlbox.find("x").text)
                ymax = int(xmlbox.find("y").text)
            # need to complement the bias
            b = [xmin, xmax, ymin, ymax]
            b = [0 if v < 100 else v - 100 for v in b]
            # print(b)
            b1, b2, b3, b4 = b
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")


# ########## train ##########
# print("Converting 'Train'")
# rgb_xml_path = "../train/trainlabel"
# rgb_txt_path = "cropped/train/trainlabel"
# ir_xml_path = "../train/trainlabelr"
# ir_txt_path = "cropped/train/trainlabelr"
# ########## val ##########
# print("Converting 'Val'")
# rgb_xml_path = '../val/vallabel'
# rgb_txt_path = 'cropped/val/vallabel'
# ir_xml_path = '../val/vallabelr'
# ir_txt_path = 'cropped/val/vallabelr'
# ########## test ##########
# print("Converting 'Test'")
# rgb_xml_path = '../test/testlabel'
# rgb_txt_path = 'cropped/test/testlabel'
# ir_xml_path = '../test/testlabelr'
# ir_txt_path = 'cropped/test/testlabelr'

for type_ in ["train", "val", "test"]:

    print(f"Converting '{type_}'")

    rgb_xml_path = f'../{type_}/{type_}label'
    rgb_txt_path = f'cropped/{type_}/{type_}label'
    ir_xml_path = f'../{type_}/{type_}labelr'
    ir_txt_path = f'cropped/{type_}/{type_}labelr'

    if not os.path.exists(rgb_xml_path):
        raise ValueError
    if not os.path.exists(ir_xml_path):
        raise ValueError

    if not os.path.exists(rgb_txt_path):
        os.makedirs(rgb_txt_path)
    if not os.path.exists(ir_txt_path):
        os.makedirs(ir_txt_path)

    for rgbxml, irxml in tzip(os.listdir(rgb_xml_path), os.listdir(ir_xml_path)):
        filename, ext = os.path.splitext(rgbxml)
        if ext == ".xml" or ext == ".XML":
            xml2yolo(rgbxml, rgb_xml_path, rgb_txt_path)

        filename, ext = os.path.splitext(irxml)
        if ext == ".xml" or ext == ".XML":
            xml2yolo(irxml, ir_xml_path, ir_txt_path)
