import unittest

from torchvision import models
import torch

from architectures import get_arch_to_constructor_dict
from classifier import FrozenFeatureDetector
from classifier import FlowerClassifier


class FeatureDetectorTests(unittest.TestCase):
    arch_dict = get_arch_to_constructor_dict()

    def setUp(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_detector = FrozenFeatureDetector(FeatureDetectorTests.arch_dict["resnet18"], self.device)

    def test_creates_correct_pretrained_detector(self):
        self.assertEqual(type(self.feature_detector.network), models.resnet.ResNet)
        fd_params = list(self.feature_detector.network.parameters())
        expected_arch = models.resnet18(pretrained=True).to(self.device)
        expected_params = list(expected_arch.parameters())
        self.assertEqual(len(fd_params), len(expected_params))
        for i in range(len(fd_params)):
            self.assertTrue(torch.all(torch.eq(fd_params[i], expected_params[i])))

    def test_detector_training_is_disabled(self):
        def is_trained(param): return param.requires_grad
        params = self.feature_detector.network.parameters()
        trained_params = list(filter(is_trained, params))
        self.assertEqual(len(trained_params), 0)

    def test_detector_is_on_selected_device(self):
        params = list(self.feature_detector.network.parameters())
        for param in params:
            self.assertEqual(param.device, self.device)


class FlowerClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.drop = 0.01
        self.classifier = FlowerClassifier("vgg13", 5000, 200, self.device, self.drop)

    def test_classifier_is_on_correct_device(self):
        params = list(self.classifier.parameters())
        for param in params:
            self.assertEqual(param.device, self.device)

    def test_classifier_is_not_frozen(self):
        params = list(self.classifier.parameters())
        for param in params:
            self.assertTrue(param.requires_grad)

    def test_classifier_has_correct_architecture(self):
        vgg13_feature_count = 25088
        expected_h1_count = 3334
        expected_h2_count = 1666
        expected_out_count = 200

        self.assertEqual(self.classifier.hidden1.in_features, vgg13_feature_count)
        self.assertEqual(self.classifier.hidden1.out_features, expected_h1_count)
        self.assertEqual(self.classifier.hidden2.in_features, expected_h1_count)
        self.assertEqual(self.classifier.hidden2.out_features, expected_h2_count)
        self.assertEqual(self.classifier.output.in_features, expected_h2_count)
        self.assertEqual(self.classifier.output.out_features, expected_out_count)
        self.assertEqual(self.classifier.dropout.p, self.drop)


if __name__ == '__main__':
    unittest.main()
