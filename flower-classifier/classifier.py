import torch
from torch import nn
from torch.nn import functional as fn
from torchvision import models

from architectures import get_arch_to_constructor_dict


class FlowerClassifier(nn.Module):
    __detector_feature_counts = {"resnet50": 2048,
                                 "resnet18": 512,
                                 "vgg13": 25088,
                                 "vgg11": 25088,
                                 "densenet121": 1024,
                                 "googlenet": 1024}

    def __init__(self, detector_architecture, hidden_unit_count, output_count, device: torch.device, dropout_prob=0.05):
        super().__init__()
        assert detector_architecture in FlowerClassifier.__detector_feature_counts.keys(), \
            f"Unknown feature detector architecture passed to classifier: {detector_architecture}"

        x_count = FlowerClassifier.__detector_feature_counts[detector_architecture]
        h1_count, h2_count = FlowerClassifier.__calculate_hidden_counts(hidden_unit_count)

        self.hidden1 = nn.Linear(x_count, h1_count)
        self.hidden2 = nn.Linear(h1_count, h2_count)
        self.output = nn.Linear(h2_count, output_count)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.to(device)

    @staticmethod
    def __calculate_hidden_counts(total_units) -> (int, int):
        h2 = total_units // 3
        h1 = total_units - h2
        return h1, h2

    def forward(self, categories):
        h1 = self.dropout(fn.relu(self.hidden1(categories)))
        h2 = self.dropout(fn.relu(self.hidden2(h1)))
        out_logits = fn.log_softmax(self.output(h2), dim=1)
        return out_logits


class FrozenFeatureDetector:
    def __init__(self, architecture: str, device: torch.device):
        arch_dict = get_arch_to_constructor_dict()
        assert architecture in arch_dict.keys(), f"Unsupported model architecture provided: {architecture}."
        self.network = arch_dict[architecture](pretrained=True)
        self.__disable_training()
        self.__move_to_device(device)
        print(f"{type(self.network)} loaded with params pre-trained and frozen.")

    def __disable_training(self):
        for parameter in self.network.parameters():
            parameter.requires_grad = False

    def __move_to_device(self, device):
        self.network.to(device)

    def attach_classifier_to_network(self, classifier: FlowerClassifier):
        architectures_with_fc_module = [models.ResNet, models.GoogLeNet]
        if type(self.network) in architectures_with_fc_module:
            self.network.fc = classifier
        else:
            self.network.classifier = classifier
        print(f"{type(classifier)} was attached to feature detector {type(self.network)}.")
        return self.network
