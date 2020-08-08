import argparse

from train_arguments import allow_positive_ints


class PredictArgumentParser:
    def __init__(self):
        self.help_dict = self.__create_arg_help_dict()
        self.arg_parser = self.__create_arg_parser()
        self.args = self.arg_parser.parse_args()

    @staticmethod
    def __create_arg_help_dict():
        help_dict = {"input": "Path to the flower image to be classified.",
                     "checkpoint": "Path to the training checkpoint of the classifier NN.",
                     "top_k": "Number of top most probable classes to return in the result.",
                     "category_names": "Path to a JSON file containing names of classified flowers.",
                     "gpu": "Causes the prediction calculations to be carried out on a GPU. "
                            "Requires CUDA support on host machine."}
        return help_dict

    def __create_arg_parser(self) -> argparse.ArgumentParser:
        arg_parser = argparse.ArgumentParser(description="Uses a pre-trained network to classify an image.")
        arg_parser.add_argument("input", help=self.help_dict["input"])
        arg_parser.add_argument("checkpoint", help=self.help_dict["checkpoint"])
        arg_parser.add_argument("--top_k", default=3, type=allow_positive_ints, help=self.help_dict["top_k"])
        arg_parser.add_argument("--category_names", default="cat_to_name.json", help=self.help_dict["category_names"])
        arg_parser.add_argument("--gpu", action="store_true", default=False, help=self.help_dict["gpu"])
        return arg_parser
