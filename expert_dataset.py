from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import json
import torch

class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root):
        self.data_root = data_root
        # Your code here
	rgbpath = self.data_root + "/rgb"
	measpath = self.data_root + "/measurements"
	rgb_names = os.listdir(rgbpath)
	meas_names = os.listdir(measpath)
	
	length = len([name for name in rgb_names])
	
	rgb = np.empty(length, 512, 512, 3)
	for idx, name in enumerate(rgb_names):
		img= cv2.imread(rgbpath + name)
		rgb[idx] = img
	
	measures = np.empty(length)
	for idx, name in enumerate(meas_names):
		measure = json.loads(measpath + name).items()
		measures[idx] = measure
	self.rgb = torch.from_numpy(rgb)
	self.measures = torch.from_numpy(measures)

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        return self.rgb[index], self.measures[index]
