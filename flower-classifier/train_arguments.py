import argparse

from architectures import get_arch_to_constructor_dict


def allow_positive_ints(value_string: str) -> int:
    number = int(value_string)
    if number <= 0:
        raise argparse.ArgumentTypeError("A positive integer is required")
    else:
        return number


def allow_positive_floats(value_string: str) -> float:
    number = float(value_string)
    if number <= 0:
        raise argparse.ArgumentTypeError("A positive float is required")
    else:
        return number


class TrainArgumentParser:
    def __init__(self):
        self.arch_dict = get_arch_to_constructor_dict()
        self.help_dict = self.__create_arg_help_dict()
        self.arg_parser = self.__create_arg_parser()
        self.args = self.arg_parser.parse_args()

    @staticmethod
    def __create_arg_help_dict():
        help_dict = {"data_dir": "Path to directory containing training data. "
                                 "Should contain 'training', 'test' and 'valid' sub-directories.",
                     "save_dir": "Path to directory in which classifier training checkpoints will be saved.",
                     "arch": "Name of pre-trained feature detector to be used. "
                             "Must be available in torchvision.models.",
                     "learning_rate": "The learning rate hyperparameter used for classifier training.",
                     "hidden_units": "Total number of hidden units in the classifier's hidden layers (summed up).",
                     "epochs": "Number of training iterations.",
                     "gpu": "Enables using GPU to perform training calculations. "
                            "Requires CUDA support on host machine."}
        return help_dict

    def __create_arg_parser(self) -> argparse.ArgumentParser:
        arg_parser = argparse.ArgumentParser(description="Trains an image classifier neural network.")
        arg_parser.add_argument("data_dir",
                                nargs='?',
                                default="flowers",
                                help=self.help_dict["data_dir"])
        arg_parser.add_argument("--save_dir",
                                default="",
                                help=self.help_dict["save_dir"])
        arg_parser.add_argument("--epochs",
                                default=5,
                                type=allow_positive_ints,
                                help=self.help_dict["epochs"])
        arg_parser.add_argument("--gpu",
                                action="store_true",
                                default=False,
                                help=self.help_dict["gpu"])
        arg_parser.add_argument("--arch",
                                default="resnet50",
                                choices=self.arch_dict.keys(),
                                help=self.help_dict["arch"])
        arg_parser.add_argument("--learning_rate",
                                default=0.0003,
                                type=allow_positive_floats,
                                help=self.help_dict["learning_rate"])
        arg_parser.add_argument("--hidden_units",
                                default=1536,
                                type=allow_positive_ints,
                                help=self.help_dict["hidden_units"])
        return arg_parser
