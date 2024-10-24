import os
import shutil
from tqdm.contrib import tzip
from tqdm import tqdm
from pathlib import Path


cropped_path = "cropped"
ir_path = "DroneVehicle/infrared"
rgb_path = "DroneVehicle/visible"

"""
Notes:
    The directory structure assumed for the 'cropped' folder:
        - cropped
            ├─ test
            │   ├─ testimg
            │   ├─ testimgr
            │   ├─ testlabel
            │   └─ testlabelr
            ├─ train
            │   ├─ trainimg
            │   ├─ trainimgr
            │   ├─ trainlabel
            │   └─ trainlabelr
            └─ val
                ├─ valimg
                ├─ valimgr
                ├─ vallabel
                └─ vallabelr


    The directory structure for output RGB and IR folder:
        - infrared/visible
            ├─ test
            │   ├─ images
            │   └─ labels
            ├─ train
            │   ├─ images
            │   └─ labels
            └─ val
                ├─ images
                └─ labels
"""

for type_ in ["train", "val", "test"]:
    for phase in ["images", "labels"]:
        if phase == "images":
            rgb_in = os.path.join(cropped_path, type_, f"{type_}img")
            ir_in = os.path.join(cropped_path, type_, f"{type_}imgr")
        elif phase == "labels":
            rgb_in = os.path.join(cropped_path, type_, f"{type_}label")
            ir_in = os.path.join(cropped_path, type_, f"{type_}labelr")
        else:
            raise ValueError

        # check input path
        if not os.path.exists(rgb_in):
            raise FileNotFoundError
        if not os.path.exists(ir_in):
            raise FileNotFoundError

        print(f"Input RGB directory: {rgb_in}")
        print(f"Input IR directory: {ir_in}")

        rgb_out = os.path.join(rgb_path, type_, phase)
        ir_out = os.path.join(ir_path, type_, phase)

        if not os.path.exists(rgb_out):
            Path(rgb_out).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(ir_out):
            Path(ir_out).mkdir(parents=True, exist_ok=True)

        print(f"Output RGB directory: {rgb_out}")
        print(f"Output IR directory: {ir_out}")

        if phase == "images":
            for rgb_item, ir_item in tzip(os.listdir(rgb_in), os.listdir(ir_in)):
                shutil.copy(os.path.join(rgb_in, rgb_item) , rgb_out)
                shutil.copy(os.path.join(ir_in, ir_item), ir_out)
        elif phase == "labels":
            for label in tqdm(os.listdir(rgb_in)):
                shutil.copy(os.path.join(rgb_in, label) , rgb_out)
                shutil.copy(os.path.join(rgb_in, label), ir_out)

