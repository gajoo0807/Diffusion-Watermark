import glob
import os
import torchvision
import torch
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset
import numpy as np
import torchvision.transforms as transforms


class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    
    def __init__(self, split, im_path, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        
        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        
        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_path)
        
        # Whether to load images and call vae or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
        
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in [1, 2]:
            data_path = os.path.join(im_path, str(d_name))
            dist_path = f'fingerprint/distribution/model_{d_name}_probabilities.pth'
            all_probabilities = torch.load(dist_path)
            # fnames: list of all image file names in the data_path
            fnames = glob.glob(os.path.join(data_path, '*.{}'.format('png')))
            fnames += glob.glob(os.path.join(data_path, '*.{}'.format('jpg')))
            fnames += glob.glob(os.path.join(data_path, '*.{}'.format('jpeg')))


            for fname in fnames:
                ims.append(fname)
                random_element = all_probabilities[np.random.randint(len(all_probabilities))]
                reshaped_element = random_element.reshape(-1)
                labels.append(reshaped_element)
                # labels.append(all_probabilities[np.random.randint(len(all_probabilities))])
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        cond_inputs = self.labels[index]
        #######################################
        
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            # shape [1, 28, 28]
            im = Image.open(self.images[index])

                    # 定义转换流程
            transform = transforms.Compose([
                # transforms.Resize((256, 256)),  # 调整图像尺寸到224x224
                transforms.Grayscale(num_output_channels=3),  # 转换为三通道灰度图
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize([0.5], [0.5])  # 标准化到[-1, 1]
            ])
            im_tensor = transform(im)
            # print(f'{im_tensor.shape=}') torch.Size([3, 28, 28])
            return im_tensor, cond_inputs