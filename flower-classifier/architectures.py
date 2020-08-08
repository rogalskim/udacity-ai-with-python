from torchvision import models


def get_arch_to_constructor_dict():
    arch_dict = {"resnet50": models.resnet50,
                 "resnet18": models.resnet18,
                 "vgg13": models.vgg13,
                 "vgg11": models.vgg11,
                 "densenet121": models.densenet121,
                 "googlenet": models.googlenet}
    return arch_dict
