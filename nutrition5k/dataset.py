import os

import numpy as np
from PIL import Image
from numpy import asarray
import pandas as pd
import torch
from skimage import transform
from torch.utils.data import Dataset
import torchvision.transforms.functional as functional


class Resize:
    """Resize the image in a sample to a given size.

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

        return {'image': img, 'mass': sample['mass'], 'calories': sample['calories']}


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = sample['image'].transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'mass': torch.from_numpy(sample['mass']),
                'calories': torch.from_numpy(sample['calories'])}


class Normalize:
    """Normalize image values."""
    def __init__(self, image_means, image_stds, mass_max=250, calories_max=200):
        self.means = image_means
        self.stds = image_stds
        self.mass_max = mass_max
        self.calories_max = calories_max

    def __call__(self, sample):
        sample['mass'] = sample['mass'] / self.mass_max
        sample['calories'] = sample['calories'] / self.calories_max
        sample['image'] = functional.normalize(sample['image'], self.means, self.stds)
        return sample


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

        frames_path = os.path.join(self.root_dir, 'imagery', 'side_angles', self.dish_calories.iloc[idx]['dish_id'],
                                   'camera_A')
        frame = os.path.join(frames_path, '1.jpg')

        image = Image.open(frame)
        image = asarray(image)

        mass = self.dish_calories.iloc[idx]['mass']
        mass = np.array([mass])
        mass = mass.astype('float').reshape(1, 1)
        calories = self.dish_calories.iloc[idx]['calories']
        calories = np.array([calories])
        calories = calories.astype('float').reshape(1, 1)
        sample = {'image': image, 'mass': mass, 'calories': calories}

        if self.transform:
            sample = self.transform(sample)

        return sample
