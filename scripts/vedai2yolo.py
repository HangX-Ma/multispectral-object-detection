# -*- coding: utf-8 -*-
import numpy as np
import os

# 1:  car
# 2:  truck
# 4:  tractor
# 5:  camping car
# 7:  motorcycle
# 8:  bus
# 9:  van
# 10: other
# 11: pickup
# 12: large
# 23: boat
# 31: plane

size = [512, 512]  # [w, h]

classdict = {
    1: 0, # car
    2: 1, # truck
    4: 2, # tractor
    5: 3, # camping car
    7: 4, # motorcycle
    8: 5, # bus
    9: 6, # van
    10: 7, # other
    11: 8, # pickup
    12: 9, # large
    23: 10, # boat
    31: 11, # plane
}


def convert(fields):
    # txt fields:
    # [0-5]: 'x_center', 'y_center', 'orientation', 'class', 'is_contained', 'is_occluded'
    # [6-9]: 'corner1_x', 'corner2_x', 'corner3_x', 'corner4_x'
    # [10-13]: 'corner1_y', 'corner2_y', 'corner3_y', 'corner4_y'

    cx = fields[0]
    cy = fields[1]
    orientation = fields[2]

    X = fields[6:10]
    Y = fields[10:14]

    l = [
        np.sqrt((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2),
        np.sqrt((X[1] - X[2]) ** 2 + (Y[1] - Y[2]) ** 2),
        np.sqrt((X[2] - X[3]) ** 2 + (Y[2] - Y[3]) ** 2),
        np.sqrt((X[3] - X[0]) ** 2 + (Y[3] - Y[0]) ** 2),
    ]

    ltri = np.sort(l)[::-1]
    longueurCar = np.mean([ltri[0], ltri[1]])
    largeurCar = np.mean([ltri[2], ltri[3]])

    ktri = np.argsort(l)[::-1]
    vecteurLongueur1 = np.array(
        [
            X[ktri[0] % 4] - X[(ktri[0] - 1) % 4],
            Y[ktri[0] % 4] - Y[(ktri[0] - 1) % 4],
        ]
    )
    vecteurLongueur2 = np.array(
        [
            X[ktri[1] % 4] - X[(ktri[1] - 1) % 4],
            Y[ktri[1] % 4] - Y[(ktri[1] - 1) % 4],
        ]
    )

    if np.dot(vecteurLongueur1, vecteurLongueur2) > 0:
        vecteurLongueur = (vecteurLongueur1 + vecteurLongueur2) / 2
    else:
        vecteurLongueur = (vecteurLongueur1 - vecteurLongueur2) / 2

    if vecteurLongueur[0] == 0:
        orientation = -np.pi / 2
    else:
        orientation = np.arctan(vecteurLongueur[1] / vecteurLongueur[0])

    pt1 = [
        cy + largeurCar / 2 + longueurCar / 2 * np.sin(-orientation),
        cx
        + largeurCar / 2 * np.sin(-orientation)
        - longueurCar / 2 * np.cos(-orientation),
    ]
    pt2 = [
        cy + largeurCar / 2 - longueurCar / 2 * np.sin(-orientation),
        cx
        + largeurCar / 2 * np.sin(-orientation)
        + longueurCar / 2 * np.cos(-orientation),
    ]
    pt3 = [
        cy - largeurCar / 2 - longueurCar / 2 * np.sin(-orientation),
        cx
        - largeurCar / 2 * np.sin(-orientation)
        + longueurCar / 2 * np.cos(-orientation),
    ]
    pt4 = [
        cy - largeurCar / 2 + longueurCar / 2 * np.sin(-orientation),
        cx
        - largeurCar / 2 * np.sin(-orientation)
        - longueurCar / 2 * np.cos(-orientation),
    ]

    x = [pt1[1], pt2[1], pt3[1], pt4[1]]
    y = [pt1[0], pt2[0], pt3[0], pt4[0]]

    sortx = np.sort(x)[::-1]
    sorty = np.sort(y)[::-1]

    xmax = sortx[0]
    xmin = sortx[3]
    ymax = sorty[0]
    ymin = sorty[3]

    W, H = size
    if xmax > W:
        xmax = W
    if xmin > W:
        xmin = W
    if ymax > H:
        ymax = H
    if ymin > H:
        ymin = H

    dw = 1.0 / W
    dh = 1.0 / H

    x = (xmax + xmin) / 2.0 - 1
    y = (ymax + ymin) / 2.0 - 1
    w = xmax - xmin
    h = ymax - ymin

    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh

    return x, y, w, h


def vedai2yolo(filename, annotpath, yolopath, count):
    ifname = str(annotpath + "/%s.txt" % filename)
    ofname = str(yolopath + "/%s.txt" % filename)

    print("{}: convert {} to {}".format(count, ifname, ofname))

    infile = open(ifname, encoding="UTF-8")
    outfile = open(ofname, "w")

    for line in infile:
        fields = line.strip().split()
        # Convert fields to appropriate data types if needed
        fields = [float(field) if "." in field else int(field) for field in fields]

        clsid = classdict[int(fields[3])]
        bbox = convert(fields)
        outfile.write(str(clsid) + " " + " ".join([str(val) for val in bbox]) + "\n")

    infile.close()
    outfile.close()


annotpath = "/hy-tmp/Annotations512"
yolopath = "/hy-tmp/Annotations"
count = 0

if not annotpath or not os.path.exists(annotpath):
    raise NotADirectoryError

if not os.path.exists(yolopath):
    os.makedirs(yolopath)

print("##### START CONVERT VEDAI TO YOLO FORMAT #####")

for annot in os.listdir(annotpath):
    count += 1
    filename, ext = os.path.splitext(annot)
    vedai2yolo(filename, annotpath, yolopath, count)

print("##### DONE {} #####".format(count))
