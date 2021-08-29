import os

import numpy as np
from PIL import Image
from numpy import asarray
import pandas as pd
import torch
import torch.nn as nn
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

import cv2


class Nutrition5kDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        csv_files = [os.path.join(root_dir, 'metadata', 'dish_metadata_cafe1.csv'),
                     os.path.join(root_dir, 'metadata', 'dish_metadata_cafe2.csv')]
        dish_metadata = {'dish_id': [], 'calories': []}
        for csv_file in csv_files:
            with open(csv_file, "r") as f:
                for line in f.readlines():
                    parts = line.split(',')
                    dish_metadata['dish_id'].append(parts[0])
                    dish_metadata['calories'].append(int(float(parts[2])))

        self.dish_calories = pd.DataFrame.from_dict(dish_metadata)

    def __len__(self):
        return len(self.dish_calories)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frames_path = os.path.join(self.root_dir, 'imagery', 'side_angles', self.dish_calories.iloc[idx]['dish_id'], 'camera_A')
        frame = os.path.join(frames_path, '0.jpg')

        image = Image.open(frame)
        image = asarray(image)

        calories = self.dish_calories.iloc[idx]['calories']
        calories = np.array([calories])
        calories = calories.astype('float').reshape(1, 1)
        sample = {'image': image, 'calories': calories}

        if self.transform:
            sample = self.transform(sample)

        return sample
