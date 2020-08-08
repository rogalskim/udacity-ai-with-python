import torch
from torch import nn
from torch import optim

from classifier import FlowerClassifier
from data_handling import DataProvider
from logger import ResultLogger


class Trainer:
    def __init__(self, model, classifier_module: FlowerClassifier, device: torch.device, logger: ResultLogger):
        self.model = model
        self.classifier = classifier_module
        self.device = device
        self.check_model_on_device(model, device)
        self.logger = logger
        self.criterion = nn.NLLLoss()
        self.optimizer = None

    def train(self, epochs, learn_rate, data_provider: DataProvider):
        self.optimizer = optim.Adam(params=self.classifier.parameters(), lr=learn_rate)
        train_loader = data_provider.train_loader
        eval_loader = data_provider.eval_loader

        hidden_units = self.classifier.hidden1.out_features + self.classifier.hidden2.out_features
        self.logger.start_logging(data_provider.batch_size, epochs, self.optimizer, learn_rate, hidden_units)

        for epoch in range(epochs):
            train_loss = self.__run_training_step(train_loader)
            eval_loss, eval_accuracy = self.__run_evaluation_step(eval_loader)
            self.logger.process_training_epoch(epoch, train_loss, eval_loss, eval_accuracy)

        self.logger.stop_logging()

    def test(self, data_provider: DataProvider):
        print("RUNNING TESTS")
        test_loader = data_provider.test_loader
        test_loss, test_accuracy = self.__run_evaluation_step(test_loader)
        print(f"Test set loss: {test_loss:10.3f}")
        print(f"Test set accuracy: {test_accuracy * 100:1.3f}%")

    @staticmethod
    def check_model_on_device(model, device):
        def is_on_device(param): return param[0].device == device
        params = list(model.parameters())
        moved_params = list(filter(is_on_device, params))
        assert len(moved_params) == len(params), "Model has not been moved to expected device!"
        print(f"Model is on device {device}.")

    def __run_training_step(self, train_loader) -> float:
        self.model.train()
        avg_loss = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            output_logits = self.model.forward(images)
            loss = self.criterion(input=output_logits, target=labels)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

        avg_loss /= len(train_loader)
        return avg_loss

    @staticmethod
    def __calculate_accuracy(logits, labels) -> float:
        predictions = torch.exp(logits)
        predicted_classes = torch.topk(predictions, k=1, dim=1)[1]
        resized_labels = labels.view(predicted_classes.shape[0], -1)
        prediction_matches = predicted_classes == resized_labels
        accuracy = torch.mean(prediction_matches.type(torch.FloatTensor))
        return accuracy.item()

    def __run_evaluation_step(self, dataset_loader) -> (float, float):
        self.model.eval()
        avg_loss = 0
        avg_accuracy = 0

        with torch.no_grad():
            for images, labels in dataset_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output_logits = self.model.forward(images)
                loss = self.criterion(input=output_logits, target=labels)
                accuracy = self.__calculate_accuracy(output_logits, labels)

                avg_loss += loss
                avg_accuracy += accuracy

        avg_loss /= len(dataset_loader)
        avg_accuracy /= len(dataset_loader)
        return avg_loss, avg_accuracy
