from pathlib import Path

from checkpoint_io import CheckpointIo
from classifier import FrozenFeatureDetector
from classifier import FlowerClassifier
from data_handling import DataProvider
from device_setup import prepare_device
from logger import ResultLogger
from train_arguments import TrainArgumentParser
from trainer import Trainer


def main():
    arg_parser = TrainArgumentParser()
    args = arg_parser.args
    data_provider = DataProvider(Path(args.data_dir))
    device = prepare_device(args.gpu)

    detector = FrozenFeatureDetector(args.arch, device)
    class_count = 102
    classifier = FlowerClassifier(args.arch, args.hidden_units, class_count, device)

    model = detector.attach_classifier_to_network(classifier)

    log_file_path = Path(args.save_dir) / Path("train_log.txt")
    logger = ResultLogger(log_file_path)
    trainer = Trainer(model, classifier, device, logger)
    trainer.train(args.epochs, args.learning_rate, data_provider)
    trainer.test(data_provider)

    checkpoint_saver = CheckpointIo()
    checkpoint_name = "flower_classifier.pth"
    checkpoint_saver.save_checkpoint(model, args.arch, classifier, trainer.optimizer,
                                     data_provider, args.save_dir, checkpoint_name)


if __name__ == "__main__":
    main()
