from pathlib import Path
import json

import pandas as pd

from checkpoint_io import CheckpointIo
from device_setup import prepare_device
from predict_arguments import PredictArgumentParser
from predictor import Predictor


def print_results(top_probs, top_classes, names: dict = None):
    print("\nPREDICTION RESULTS")
    print("------------------\n")

    pd_probs = pd.Series(data=top_probs)
    pd_classes = pd.Series(data=top_classes)

    pd_frame_dict = {"Prob. [%]": pd_probs*100,
                     "Class": pd_classes}
    if names:
        class_names = [names[class_id].title() for class_id in top_classes]
        pd_names = pd.Series(data=class_names)
        pd_frame_dict["Name"] = pd_names

    pd_results = pd.DataFrame(pd_frame_dict)
    print(pd_results)


def get_class_names(json_path: str):
    if not Path(json_path).exists():
        print(f"Warning: failed to load class names from '{json_path}'.")
        return None
    with open(str(json_path), 'r') as file:
        name_dict = json.load(file)
    return name_dict


def main():
    arg_parser = PredictArgumentParser()
    args = arg_parser.args

    device = prepare_device(args.gpu)

    checkpoint_loader = CheckpointIo()
    model = checkpoint_loader.load_checkpoint(Path(args.checkpoint), device)

    predictor = Predictor(device)
    top_probs, top_classes = predictor.predict(args.input, model, args.top_k)
    class_names = get_class_names(args.category_names)
    print_results(top_probs, top_classes, class_names)


if __name__ == "__main__":
    main()
