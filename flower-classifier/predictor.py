import math
from pathlib import Path

import torch

from data_handling import InputImageProcessor


class Predictor:
    def __init__(self, device: torch.device):
        self.device = device

    def predict(self, image_path, model, topk):
        print(f"\nCalculating prediction for '{image_path}'...")
        input_minibatch = self.__prepare_input_minibatch(image_path, self.device)
        model.eval()
        with torch.no_grad():
            log_probabilities = model.forward(input_minibatch)
        probabilities = torch.exp(log_probabilities)

        self.__check_result(probabilities)

        top_probs, top_class_ids = torch.topk(probabilities, k=topk, dim=1)
        top_prob_values = self.__extract_values(top_probs)
        top_classes = self.__convert_ids_to_classes(top_class_ids, model)

        return top_prob_values, top_classes

    @staticmethod
    def __prepare_input_minibatch(image_path, device) -> torch.Tensor:
        image_processor = InputImageProcessor()
        input_tensor = image_processor.load_image(Path(image_path))
        input_minibatch = input_tensor.view((1, *input_tensor.shape))
        minibatch_on_device = input_minibatch.to(device)
        return minibatch_on_device

    @staticmethod
    def __extract_values(tensor: torch.Tensor) -> list:
        as_ndarray = tensor.cpu().detach().numpy()
        as_list = list(as_ndarray.flat)
        return as_list

    @staticmethod
    def __convert_ids_to_classes(ids: torch.Tensor, model) -> list:
        id_values = Predictor.__extract_values(ids)
        classes = [model.index_to_class[i] for i in id_values]
        return classes

    @staticmethod
    def __check_result(probabilities):
        prob_sum = torch.sum(probabilities)
        assert math.isclose(prob_sum, 1.0, rel_tol=1e-6), f"Probabilities don't sum up to 1, but to {prob_sum}."
