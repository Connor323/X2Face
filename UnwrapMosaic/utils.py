###################################################################################################
# Helper functions for training/testing 
# Author: Hans
# E-mail: hao66@purdue.edu
# Creating date: 07/23/2019
###################################################################################################
from torch.utils.data.dataloader import default_collate
from PIL import Image
import numpy as np
import os 
from os import path as osp
import torch
import torchvision.utils as vutils
import random

def my_collate(batch):
    filtered_batch = []
    for item in batch: 
        bad = False
        for v in item:
            if v is None: 
                bad = True
        if not bad: 
            filtered_batch.append(item)
    return default_collate(filtered_batch)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_images(visuals, image_path, save_path):
    """Save images to the disk.
    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string for creating image paths
        save_path (str)          -- the string is used to create saving paths
    """
    mkdir(save_path)
    name = osp.basename(image_path)
    name = osp.splitext(name)[0]
    for label, im_data in visuals.items():
        image_name = '%s_%s.jpg' % (name, label)
        save_image_path = os.path.join(save_path, image_name)
        vutils.save_image(im_data, save_image_path, normalize=True, scale_each=True)
        # im = tensor2im(im_data)
        # save_image(im, save_image_path)

def seed_everything(SEED=0):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED']=str(SEED)
    # torch.backends.cudnn.benchmark = False