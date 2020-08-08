import unittest
from pathlib import Path

import torch
from torchvision import transforms as tr

from data_handling import DataProvider


class DataProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_provider = DataProvider(Path("../flowers"))

    def test_creates_correct_dir_paths(self):
        self.assertEqual(self.data_provider.train_dir, Path("../flowers/train"))
        self.assertEqual(self.data_provider.test_dir, Path("../flowers/test"))
        self.assertEqual(self.data_provider.valid_dir, Path("../flowers/valid"))

    def test_provides_expected_train_data_transformation(self):
        self.assertEqual(type(self.data_provider.train_transform), tr.Compose)
        composed_transform_types = list(map(type, self.data_provider.train_transform.transforms))
        self.assertIn(tr.RandomRotation, composed_transform_types)
        self.assertIn(tr.RandomResizedCrop, composed_transform_types)
        self.assertIn(tr.RandomVerticalFlip, composed_transform_types)
        self.assertIn(tr.Normalize, composed_transform_types)
        self.assertIn(tr.ToTensor, composed_transform_types)

    def test_uses_required_image_size_in_training_data(self):
        resize_transform = next(filter(lambda t: type(t) == tr.RandomResizedCrop,
                                       self.data_provider.train_transform.transforms))
        self.assertEqual(type(resize_transform), tr.RandomResizedCrop)
        self.assertEqual(resize_transform.size, (224, 224))

    def test_provides_expected_test_data_transformation(self):
        self.assertEqual(type(self.data_provider.test_transform), tr.Compose)
        composed_transform_types = list(map(type, self.data_provider.test_transform.transforms))
        self.assertIn(tr.Resize, composed_transform_types)
        self.assertIn(tr.CenterCrop, composed_transform_types)
        self.assertIn(tr.Normalize, composed_transform_types)
        self.assertIn(tr.ToTensor, composed_transform_types)

    def test_provides_expected_test_data_transformation(self):
        self.assertEqual(type(self.data_provider.eval_transform), tr.Compose)
        composed_transform_types = list(map(type, self.data_provider.eval_transform.transforms))
        self.assertIn(tr.Resize, composed_transform_types)
        self.assertIn(tr.CenterCrop, composed_transform_types)
        self.assertIn(tr.Normalize, composed_transform_types)
        self.assertIn(tr.ToTensor, composed_transform_types)

    def test_train_loader_loads_correct_data(self):
        images, labels = next(iter(self.data_provider.train_loader))
        self.assertEqual(images.shape, torch.Size([self.data_provider.batch_size, 3, 224, 224]))
        self.assertEqual(labels.shape, torch.Size([self.data_provider.batch_size]))

    def test_test_loader_loads_correct_data(self):
        images, labels = next(iter(self.data_provider.test_loader))
        self.assertEqual(images.shape, torch.Size([self.data_provider.batch_size, 3, 224, 224]))
        self.assertEqual(labels.shape, torch.Size([self.data_provider.batch_size]))

    def test_eval_loader_loads_correct_data(self):
        images, labels = next(iter(self.data_provider.eval_loader))
        self.assertEqual(images.shape, torch.Size([self.data_provider.batch_size, 3, 224, 224]))
        self.assertEqual(labels.shape, torch.Size([self.data_provider.batch_size]))

    def test_has_cat_to_name_dict(self):
        self.assertEqual(self.data_provider.cat_to_name["28"], "stemless gentian")


if __name__ == '__main__':
    unittest.main()
