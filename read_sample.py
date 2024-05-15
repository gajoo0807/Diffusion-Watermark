import glob
import os
import torchvision
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import numpy as np


def load_image(im_path):
    ims = []
    for d_name in [1, 2]:
        data_path = os.path.join(im_path, str(d_name))
        fnames = glob.glob(os.path.join(data_path, '*.{}'.format('png')))
        fnames += glob.glob(os.path.join(data_path, '*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(data_path, '*.{}'.format('jpeg')))

        for fname in fnames:
            ims.append(fname)

    return ims


def get_item(ims, index):
    im = Image.open(ims[index])
    im_tensor = torchvision.transforms.ToTensor()(im)
    print(im_tensor)


    # Convert im_tensor back to image and save as sample.png
    im_restored = torchvision.transforms.ToPILImage()(im_tensor)
    im_restored.save('sample.png')
    return im_tensor

if __name__ == '__main__':
    im_path = 'data/mnist/train/images'
    ims = load_image(im_path)
    get_item(ims, -1)