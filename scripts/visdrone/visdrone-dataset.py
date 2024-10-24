import os
import random
from tqdm.contrib import tzip


def build_dataset(rgb_path, ir_path):
    print("###### START BUILDING DATASET ######")

    for type_ in ["train", "val", "test"]:
        rgbimgs_path = os.path.join(rgb_path, type_, "images")
        irimgs_path = os.path.join(ir_path, type_, "images")

        # check source images directory
        if not rgbimgs_path or not os.path.exists(rgbimgs_path):
            raise NotADirectoryError
        if not irimgs_path or not os.path.exists(irimgs_path):
            raise NotADirectoryError

        print(f"RGB images path: {rgbimgs_path}")
        print(f"IR  images path: {irimgs_path}")

        rgbimgs = []
        irimgs = []
        # TODO: avoid inorder sequence
        for img in sorted(os.listdir(rgbimgs_path)):
            rgbimgs.append(img)

        for img in sorted(os.listdir(irimgs_path)):
            irimgs.append(img)
        data = list(zip(rgbimgs, irimgs))

        # shuffle source files and unbind them
        random.shuffle(data)
        rgbimgs, irimgs = list(zip(*data))

        # divide source files
        rgb_txt = open(os.path.join(rgb_path, type_) + f"/{type_}.txt", "w+")
        ir_txt = open(os.path.join(ir_path, type_) + f"/{type_}.txt", "w+")

        # RGB
        for rgb, ir in tzip(rgbimgs, irimgs):
            path = os.path.join(rgbimgs_path, rgb) + "\n"
            rgb_txt.write(path)

            path = os.path.join(irimgs_path, ir) + "\n"
            ir_txt.write(path)

        os.fsync(rgb_txt)
        os.fsync(ir_txt)

        rgb_txt.close()
        ir_txt.close()


    print("###### DONE ######")


if __name__ == "__main__":
    rgb_path = "/hy-tmp/DroneVehicle/visible"
    ir_path = "/hy-tmp/DroneVehicle/infrared"

    if not os.path.exists(rgb_path):
        raise NotADirectoryError
    if not os.path.exists(ir_path):
        raise NotADirectoryError

    build_dataset(rgb_path, ir_path)
