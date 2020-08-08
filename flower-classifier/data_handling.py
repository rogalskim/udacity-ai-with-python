from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms as tr
from torch.utils.data import DataLoader as DataLoader


class ImageProperties:
    def __init__(self, size: int, resize_size: int, norm_means: list, norm_stds: list):
        self.size = size
        self.resize_size = resize_size
        self.norm_means = norm_means
        self.norm_stds = norm_stds


class DataProvider:
    def __init__(self, data_root_dir: Path):
        self.__build_paths(data_root_dir)
        self.__create_transforms()
        self.batch_size = 64
        self.__create_train_loader()
        self.__create_test_loader()
        self.__create_eval_loader()

    @staticmethod
    def get_image_properties():
        return ImageProperties(size=224,
                               resize_size=224 + 16,
                               norm_means=[0.485, 0.456, 0.406],
                               norm_stds=[0.229, 0.224, 0.225])

    def __build_paths(self, data_root_dir: Path):
        self.data_root_dir = data_root_dir
        self.train_dir = data_root_dir / Path("train")
        self.valid_dir = data_root_dir / Path("valid")
        self.test_dir = data_root_dir / Path("test")

    def __create_transforms(self):
        image_properties = self.get_image_properties()
        image_size = image_properties.size
        resize_size = image_properties.resize_size
        norm_means = image_properties.norm_means
        norm_stds = image_properties.norm_stds

        train_transforms = [tr.RandomRotation(degrees=10),
                            tr.RandomVerticalFlip(p=0.1),
                            tr.RandomResizedCrop(size=image_size, scale=(0.5, 1.5)),
                            tr.ToTensor(),
                            tr.Normalize(norm_means, norm_stds)]
        self.train_transform = tr.Compose(train_transforms)

        test_transforms = [tr.Resize(size=resize_size),
                           tr.CenterCrop(size=image_size),
                           tr.ToTensor(),
                           tr.Normalize(norm_means, norm_stds)]
        self.test_transform = tr.Compose(test_transforms)
        self.eval_transform = self.test_transform

    def __create_train_loader(self):
        self.train_data = tv.datasets.ImageFolder(self.train_dir, self.train_transform)
        self.train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True)

    def __create_test_loader(self):
        self.test_data = tv.datasets.ImageFolder(self.test_dir, self.test_transform)
        self.test_loader = DataLoader(self.test_data, self.batch_size, shuffle=True)

    def __create_eval_loader(self):
        self.eval_data = tv.datasets.ImageFolder(self.valid_dir, self.eval_transform)
        self.eval_loader = DataLoader(self.eval_data, self.batch_size, shuffle=True)


class InputImageProcessor:
    def __init__(self):
        self.properties = DataProvider.get_image_properties()

    def load_image(self, image_path: Path) -> torch.Tensor:
        assert image_path.exists(), "Input image doesn't exist!"
        image = Image.open(image_path)
        resized = self.__resize(image)
        cropped = self.__crop_image_center(resized)
        image_array = self.__image_to_ndarray(cropped)
        normalized_array = self.__normalize_image_array(image_array)
        return torch.Tensor(normalized_array)

    def __resize(self, image: Image) -> Image:
        width, height = image.size
        if height > width:
            image.thumbnail((self.properties.resize_size, height))
        else:
            image.thumbnail((width, self.properties.resize_size))
        return image

    @staticmethod
    def __find_center_coords(image: Image) -> (float, float):
        width, height = image.size
        x = width / 2
        y = height / 2
        return x, y

    def __crop_image_center(self, image: Image) -> Image:
        center_x, center_y = self.__find_center_coords(image)
        left = center_x - self.properties.size / 2
        top = center_y - self.properties.size / 2
        right = left + self.properties.size
        bottom = top + self.properties.size
        assert left >= 0 and top >= 0
        return image.crop((left, top, right, bottom))

    @staticmethod
    def __image_to_ndarray(image: Image) -> np.array:
        arr = np.array(image)
        return arr.transpose((2, 0, 1))

    def __normalize_image_array(self, image_array: np.array) -> np.array:
        assert image_array.shape == (3, self.properties.size, self.properties.size), \
            "Input array doesn't have the required shape."
        normalized_array = np.ones(image_array.shape)
        for i in range(3):
            color_channel = image_array[i] / 255
            channel_mean = self.properties.norm_means[i]
            channel_std = self.properties.norm_stds[i]
            normalized_array[i] = (color_channel - channel_mean) / channel_std
        return normalized_array

