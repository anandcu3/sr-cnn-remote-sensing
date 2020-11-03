from PIL import Image
import numpy as np
import torch

def apply_bicubic_interpolation(image_array, out_width, out_height):
    #print(image_array.astype(np.uint8))
    img = Image.fromarray(image_array.astype(np.uint8))
    return np.asarray(img.resize((out_width, out_height), Image.BICUBIC))
