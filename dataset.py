import glob
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from skimage import io
import utils
import numpy as np
import torch
from skimage import transform

class UCMercedLandUseDataset(Dataset):
    """UC Merced Land Use dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all subdirectories with the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.land_images_list = np.array(list(glob.glob(root_dir + "/Images/*/*.tif")))

    def __len__(self):
        return len(self.land_images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.land_images_list[idx]
        orig_image = io.imread(img_name, plugin='matplotlib')
        if orig_image.shape[0] != 256 or orig_image.shape[1] != 256:
            orig_image = transform.resize(orig_image,(256,256))
            orig_image = (orig_image * 255).astype(np.uint8)
        reduced_image = (transform.resize(orig_image,(256/4,256/4)) * 255).astype(np.uint8)
        orig_image = orig_image.astype(np.float32)/255
        downscaled = utils.apply_bicubic_interpolation(reduced_image, int(orig_image.shape[0]), int(orig_image.shape[1])).astype(np.float32)/255
        return downscaled, orig_image
