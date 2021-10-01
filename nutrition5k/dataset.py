import os
import random
from glob import glob

import numpy as np
from PIL import Image
from numpy import asarray
import pandas as pd
import torch
from skimage import transform
from torch.utils.data import Dataset
import torchvision.transforms as transforms
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
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        sample['image'] = transform.resize(sample['image'], (new_h, new_w), preserve_range=True).astype('uint8')

        return sample


class CenterCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        sample['image'] = functional.center_crop(sample['image'], self.output_size)
        return sample


class RandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability
        self.flip = transforms.RandomHorizontalFlip(p=probability)

    def __call__(self, sample):
        sample['image'] = self.flip(sample['image'])
        return sample


class RandomVerticalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability
        self.flip = transforms.RandomVerticalFlip(p=probability)

    def __call__(self, sample):
        sample['image'] = self.flip(sample['image'])
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample['image'] = torch.from_numpy(sample['image'].transpose((2, 0, 1)).astype(float))
        keys = set(sample.keys()).difference(['image'])
        for key in keys:
            sample[key] = torch.from_numpy(sample[key])
        return sample


class Normalize:
    """Normalize values."""

    def __init__(self, image_means, image_stds, mass_max=1.0, calories_max=1.0):
        self.means = image_means
        self.stds = image_stds
        self.mass_max = mass_max
        self.calories_max = calories_max

    def __call__(self, sample):
        sample['mass'] = sample['mass'] / self.mass_max
        sample['calorie'] = sample['calorie'] / self.calories_max
        sample['image'] = functional.normalize(sample['image'], self.means, self.stds)
        return sample


def create_nutrition_df(root_dir, sampling_rate=5):
    csv_file = os.path.join(root_dir, 'metadata', 'dish_metadata_cafe1.csv')
    dish_metadata = {'dish_id': [], 'mass': [], 'calorie': [], 'fat': [], 'carb': [], 'protein': [], 'frame': []}
    with open(csv_file, "r") as f:
        for line in f.readlines():
            parts = line.split(',')

            dish_id = parts[0]
            frames_path = os.path.join(root_dir, 'imagery', 'side_angles',
                                       dish_id,
                                       'frames')
            if not os.path.isdir(frames_path):
                continue

            frames = sorted(glob(frames_path + os.path.sep + '*.jpeg'))
            for i, frame in enumerate(frames):
                if i % sampling_rate == 0:
                    dish_metadata['dish_id'].append(parts[0])
                    dish_metadata['calorie'].append(int(float(parts[1])))
                    dish_metadata['mass'].append(parts[2])
                    dish_metadata['fat'].append(parts[3])
                    dish_metadata['carb'].append(parts[4])
                    dish_metadata['protein'].append(parts[5])
                    dish_metadata['frame'].append(frame)

    return pd.DataFrame.from_dict(dish_metadata)


def split_dataframe(dataframe: pd.DataFrame, split):
    dish_ids = dataframe.dish_id.unique()
    random.shuffle(dish_ids)
    train_end = int(len(dish_ids) * split['train'])
    train_ids = dish_ids[:train_end]
    train_df = dataframe[dataframe['dish_id'].isin(train_ids)]
    train_index = list(train_df.index.copy())
    random.shuffle(train_index)
    train_df = train_df.loc[train_index]

    val_end = train_end + int(len(dish_ids) * split['validation'])
    val_ids = dish_ids[train_end:val_end]
    val_df = dataframe[dataframe['dish_id'].isin(val_ids)]
    val_index = list(val_df.index.copy())
    random.shuffle(val_index)
    val_df = val_df.loc[val_index]

    test_ids = dish_ids[val_end:]
    test_df = dataframe[dataframe['dish_id'].isin(test_ids)]
    test_index = list(test_df.index.copy())
    random.shuffle(test_index)
    test_df = test_df.loc[test_index]

    return train_df, val_df, test_df


def to_ndarray(value):
    value = np.array([value])
    return value.astype('float').reshape(1, 1)


class Nutrition5kDataset(Dataset):
    def __init__(self, dish_metadata, root_dir, transform=None):
        self.dish_metadata = dish_metadata
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dish_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = self.dish_metadata.iloc[idx]['frame']

        image = Image.open(frame)
        image = asarray(image)

        calories = to_ndarray(self.dish_metadata.iloc[idx]['calorie'])
        mass = to_ndarray(self.dish_metadata.iloc[idx]['mass'])
        fat = to_ndarray(self.dish_metadata.iloc[idx]['fat'])
        carb = to_ndarray(self.dish_metadata.iloc[idx]['carb'])
        protein = to_ndarray(self.dish_metadata.iloc[idx]['protein'])

        sample = {'image': image, 'mass': mass, 'fat': fat, 'carb': carb, 'protein': protein, 'calorie': calories}

        if self.transform:
            sample = self.transform(sample)

        return sample
