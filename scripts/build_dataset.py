import os
import random
import argparse


def build_dataset(rgb_path, ir_path, train_rate=0.8, val_rate=0.1, test_rate=0.1):
    rate = train_rate + val_rate + test_rate
    if rate > 1.0 or rate < 0.0:
        raise ValueError

    rgbimgs_path = rgb_path + "/images"
    irimgs_path = ir_path + "/images"

    print("RGB images path: %s" % rgbimgs_path)
    print("IR  images path: %s" % irimgs_path)

    # check source images directory
    if not rgbimgs_path or not os.path.exists(rgbimgs_path):
        raise NotADirectoryError
    if not irimgs_path or not os.path.exists(irimgs_path):
        raise NotADirectoryError

    rgbimgs = []
    irimgs = []
    # TODO: avoid inorder sequence
    for img in sorted(os.listdir(rgbimgs_path)):
        rgbimgs.append(img)

    for img in sorted(os.listdir(irimgs_path)):
        irimgs.append(img)
    data = list(zip(rgbimgs, irimgs))
    n = len(rgbimgs)

    # shuffle source files and unbind them
    random.shuffle(data)
    rgbimgs, irimgs = list(zip(*data))

    # divide source files
    train_rgbimgs = rgbimgs[0 : int(train_rate * n)]
    val_rgbimgs = rgbimgs[int(train_rate * n) : int((train_rate + val_rate) * n)]
    test_rgbimgs = rgbimgs[int((train_rate + val_rate) * n) :]

    train_irimgs = irimgs[0 : int(train_rate * n)]
    val_irimgs = irimgs[int(train_rate * n) : int((train_rate + val_rate) * n)]
    test_irimgs = irimgs[int((train_rate + val_rate) * n) :]

    # RGB
    rgb_train_txt = open(rgb_path + "/train.txt", "w")
    rgb_val_txt = open(rgb_path + "/val.txt", "w")
    rgb_test_txt = open(rgb_path + "/test.txt", "w")

    # IR
    ir_train_txt = open(ir_path + "/train.txt", "w")
    ir_val_txt = open(ir_path + "/val.txt", "w")
    ir_test_txt = open(ir_path + "/test.txt", "w")

    # RGB
    for img in train_rgbimgs:
        path = rgbimgs_path + "/" + img + '\n'
        rgb_train_txt.write(path)

    for img in val_rgbimgs:
        path = rgbimgs_path + "/" + img + '\n'
        rgb_val_txt.write(path)

    for img in test_rgbimgs:
        path = rgbimgs_path + "/" + img + '\n'
        rgb_test_txt.write(path)

    # IR
    for img in train_irimgs:
        path = irimgs_path + "/" + img + '\n'
        ir_train_txt.write(path)

    for img in val_irimgs:
        path = irimgs_path + "/" + img + '\n'
        ir_val_txt.write(path)

    for img in test_irimgs:
        path = irimgs_path + "/" + img + '\n'
        ir_test_txt.write(path)

    rgb_train_txt.close()
    rgb_val_txt.close()
    rgb_test_txt.close()

    ir_train_txt.close()
    ir_val_txt.close()
    ir_test_txt.close()

    print("###### DONE ######")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", default="rgb", type=str, help="RGB set path")
    parser.add_argument("--ir_path", default="ir", type=str, help="IR set path")
    opt = parser.parse_args()

    rgb_path = opt.rgb_path
    ir_path = opt.ir_path

    # rgb_path = "/hy-tmp/align/visible"
    # ir_path = "/hy-tmp/align/infrared"
    rgb_path = "/hy-tmp/DroneVehicle/visible"
    ir_path = "/hy-tmp/DroneVehicle/infrared"

    print("###### START BUILDING DATASET ######")
    print("RGB root path: %s" % rgb_path)
    print("IR  root path: %s" % ir_path)

    build_dataset(rgb_path, ir_path, train_rate=0.8, val_rate=0.1, test_rate=0.1)
