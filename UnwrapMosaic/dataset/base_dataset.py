###################################################################################################
# The base class for dataset.
# This class comes from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/base_dataset.py 
# Author: Hans
# E-mail: hao66@purdue.edu
# Creating date: 07/21/2019
###################################################################################################
from torch.utils.data.dataset import Dataset

class BaseDataset(Dataset):
    """This class is an abstract base class for datasets.
    """
    def __init__(self, opt):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.imsize = opt.image_size
        self.isTrain = opt.isTrain
        self.transform = None 

    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass
    
    def get_files(self):
        """Find file names to read later. This has no return value. 
        """
        pass
    
    def get_transform(self):
        """get the trainsform function
        """
        pass 