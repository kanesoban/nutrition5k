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


class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'calories': sample['calories']}


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, calories = sample['image'], sample['calories']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'calories': torch.from_numpy(calories)}


class Nutrition5kDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        csv_files = [os.path.join(root_dir, 'metadata', 'dish_metadata_cafe1.csv'),
                     os.path.join(root_dir, 'metadata', 'dish_metadata_cafe2.csv')]
        dish_metadata = {'dish_id': [], 'mass': [], 'calories': []}
        for csv_file in csv_files:
            with open(csv_file, "r") as f:
                for line in f.readlines():
                    parts = line.split(',')
                    dish_metadata['dish_id'].append(parts[0])
                    dish_metadata['mass'].append(parts[1])
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
