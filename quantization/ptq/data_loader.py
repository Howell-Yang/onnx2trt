#!/usr/bin/python3
# encoding:utf-8
import random
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class DataLoader:
    def __init__(self, image_dir="/mapai/howellyang/code/onnx2trt/calib_images_2k"):
        self.image_dir = image_dir

    def get_numpy_data(self, image_num=100):
        calibration_files = glob.glob(
            os.path.join(self.image_dir, "*.jpg")
        )
        calibration_files += glob.glob(
            os.path.join(self.image_dir, "*.png")
        )
        random.seed(103600)
        random.shuffle(calibration_files)
        calibration_files = calibration_files[:image_num]
        input_data = [DataLoader.convert_any_to_numpy(self.read_image(path))
                      for path in calibration_files]
        return input_data

    def read_image(self, path):
        _img_transforms = transforms.Compose(
            [
                transforms.Resize((384, 768)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ]
        )
        img = Image.open(path).convert("RGB")
        img_w, img_h = img.size[0], img.size[1]
        img = _img_transforms(img)
        img = img.unsqueeze(0)
        return img

    @staticmethod
    def convert_any_to_numpy(x, accepet_none: bool = True) -> np.ndarray:
        if x is None and not accepet_none:
            raise ValueError("Trying to convert an empty value.")
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, int) or isinstance(x, float):
            return np.array(
                [
                    x,
                ]
            )
        elif isinstance(x, torch.Tensor):
            if x.numel() == 0 and accepet_none:
                return None
            if x.numel() == 0 and not accepet_none:
                raise ValueError("Trying to convert an empty value.")
            if x.numel() == 1:
                return DataLoader.convert_any_to_numpy(x.detach().cpu().item())
            if x.numel() > 1:
                return x.detach().cpu().numpy()
        elif isinstance(x, list) or isinstance(x, tuple):
            return np.array(x)
        else:
            raise TypeError(
                f"input value {x}({type(x)}) can not be converted as numpy type."
            )


def read_image(path):
    # 多任务模型
    _img_transforms = transforms.Compose(
        [
            transforms.Resize((384, 768)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = Image.open(path).convert("RGB")
    img_w, img_h = img.size[0], img.size[1]
    img = _img_transforms(img)
    img = img.unsqueeze(0)
    return img
