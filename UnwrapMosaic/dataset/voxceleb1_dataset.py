###################################################################################################
# Dataset reader for Voxceleb1 dataset 
# Author: Hans
# E-mail: hao66@purdue.edu
# Creating date: 07/22/2019
###################################################################################################
import cv2
import random
import torch 
from .base_dataset import BaseDataset
from torchvision import transforms
import numpy as np 
import os 
from os import path as osp 
import glob 
from PIL import Image
from sklearn.model_selection import train_test_split

class Voxceleb1Dataset:
    """
    Read Voxceleb1 dataset. 
    """
    def __init__(self, dataroot, image_size, isTrain):
        self.root = dataroot
        self.imsize = image_size
        self.isTrain = isTrain
        self.num_videos = 0
        self.get_files()

    def get_files(self):
        """
        Split the identity into training/testing
        """
        identities = os.listdir(self.root)
        if self.isTrain: 
            self.identities = train_test_split(identities, train_size=0.8, random_state=0)[0]
        else:
            self.identities = train_test_split(identities, train_size=0.8, random_state=0)[1]
        
        vid_num = 0
        for person_id in self.identities:
            for video_id in os.listdir(os.path.join(self.root, person_id)):
                for video in os.listdir(os.path.join(self.root, person_id, video_id)):
                    vid_num += 1
        self.num_videos = vid_num
    
    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return self.num_videos
    
    def __getitem__(self, idx):
        """
        Return a data point and its metadata information.
        Parameters:
            idx - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        vid_idx = idx
        if idx<0:
            idx = self.__len__() + idx
        for person_id in self.identities:
            for video_id in os.listdir(os.path.join(self.root, person_id)):
                for video in os.listdir(os.path.join(self.root, person_id, video_id)):
                    if idx != 0:
                        idx -= 1
                    else:
                        break
                if idx == 0:
                    break
            if idx == 0:
                break
        path = os.path.join(self.root, person_id, video_id, video)
        frames_list, path_list = select_frames_new(path, 2, self.imsize)

        frames = torch.from_numpy(np.array(frames_list)).type(dtype = torch.float) #2,224,224,3
        frames = frames.transpose(1,3) #2,3,224,224
        frames = frames.transpose(2,3) #2,3,224,224

        embedder_input = frames[0].squeeze() # target landmark
        generator_input = frames[1].squeeze() # original appearance 
        generator_gt = frames[0].squeeze() # target appearance
        query_file_path = path_list[0]
        return embedder_input, generator_input, generator_gt, query_file_path


def select_frames_new(video_path, K, imsize=224):
    files_tmp = sorted(glob.glob(osp.join(video_path, "*.jpg")))
    
    txt_path = osp.join(video_path, "landmarks.txt")
    names_with_landmark = set()
    with open(txt_path, "r") as file_reader: 
        for line in file_reader.readlines(): 
            name = line.split(",")[0]
            names_with_landmark.add(name)
    files = []
    for f in files_tmp: 
        name = osp.basename(f)
        if name in names_with_landmark: 
            files.append(f)

    n_frames = len(files)    
    if n_frames <= K: #There are not enough frames in the video
        rand_frames_idx = [1]*n_frames
    else:
        rand_frames_idx = [0]*n_frames
        i = 0
        while(i < K):
            idx = random.randint(0, n_frames-1)
            if rand_frames_idx[idx] == 0:
                rand_frames_idx[idx] = 1
                i += 1
    
    frames_list = []
    path_list = []
    
    # Read until video is completed or no frames needed
    frame_idx = 0
    while(len(frames_list) < K):
        if len(frames_list) >= n_frames: 
            index = np.random.randint(len(frames_list))
            rand_frame = frames_list[index].copy()
            frames_list.append(rand_frame)
            path_list.append(path_list[index])
        elif rand_frames_idx[frame_idx] == 1:
            frame = cv2.imread(files[frame_idx])
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            RGB = cv2.resize(RGB, (imsize, imsize))
            frames_list.append(RGB)
            path_list.append(video_path)
        frame_idx += 1
    return frames_list, path_list