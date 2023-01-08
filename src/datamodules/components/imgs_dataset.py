import os.path
import random

import cv2
import numpy as np
from torch.utils import data
from scipy import signal

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    print(os.path.abspath(dir))
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImagesDataset(data.Dataset):
    """Dataset without order
    Args:
        transforms (Callable):
        dataroot (str): path to dataset
        phase (str): phase for folders
        direction (str): dataset
        input_nc (int): channels for A or B dataset (depend on direction)
        output_nc (int): channels for B or A dataset (depend on direction)
    """

    def __init__(self, transforms, dataroot, input_nc=3, output_nc=3):
        self.root = dataroot
        self.img_dir = dataroot
        self.img_paths = make_dataset(self.img_dir)
        self.img_paths = sorted(self.img_paths)

        self.imgs_count = len(self.img_paths)

        self.transform = transforms

        self.input_nc = input_nc
        self.output_nc = output_nc

    def __getitem__(self, index):
        img_path = self.img_paths[index % self.imgs_count]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augment_image = self.transform(image=img)['image']

        return dict(image=augment_image, idx=index)

    def __len__(self):
        return self.imgs_count
