from pathlib import Path

import torch

from classifier import FlowerClassifier
from classifier import FrozenFeatureDetector


class CheckpointIo:
    def save_checkpoint(self, model, model_arch, classifier, optimizer, data_provider, save_dir, file_name):
        save_path = Path(save_dir) / Path(file_name)
        model_dict = self.__serialize_model(model, model_arch, classifier, optimizer, data_provider.eval_data)
        torch.save(model_dict, save_path)
        print(f"\nModel saved to '{save_path}'.")

    def load_checkpoint(self, file_path: Path, device: torch.device):
        assert file_path.exists(), "Model checkpoint file doesn't exist!"
        print(f"\nLoading model from '{file_path}'...")
        model_dict = torch.load(file_path)

        feature_detector = self.__deserialize_feature_detector(model_dict, device)
        classifier = self.__deserialize_classifier(model_dict, device)

        model = feature_detector.attach_classifier_to_network(classifier)
        state_dict = model_dict["model state"]
        model.load_state_dict(state_dict)
        model.to(device)
        model.index_to_class = self.__get_index_to_class_dict(model_dict)
        print(f"Successfully loaded model architecture and state.")

        return model

    @staticmethod
    def __serialize_model(model, model_arch: str, classifier: FlowerClassifier, optimizer, eval_data) -> dict:
        return {"hidden units": classifier.hidden1.out_features + classifier.hidden2.out_features,
                "output units": classifier.output.out_features,
                "dropout prob": classifier.dropout.p,
                "model state": model.state_dict(),
                "feature detector": model_arch,
                "class_to_index": eval_data.class_to_idx,
                "optimizer state": optimizer.state_dict()}

    @staticmethod
    def __deserialize_feature_detector(model_dict, device) -> FrozenFeatureDetector:
        architecture = model_dict["feature detector"]
        return FrozenFeatureDetector(architecture, device)

    @staticmethod
    def __deserialize_classifier(model_dict, device) -> FlowerClassifier:
        architecture = model_dict["feature detector"]
        hidden_units = model_dict["hidden units"]
        output_units = model_dict["output units"]
        dropout = model_dict["dropout prob"]
        return FlowerClassifier(architecture, hidden_units, output_units, device, dropout)

    @staticmethod
    def __get_index_to_class_dict(model_dict):
        class2index = model_dict["class_to_index"]
        index_to_class = {value: key for key, value in class2index.items()}
        return index_to_class
