from datetime import datetime
from pathlib import Path

# noinspection SpellCheckingInspection
class ResultLogger:
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.file = None
        self.start_time = None

    @staticmethod
    def __format_hyperparams(batch_size, epoch_count, optimizer, lr, hidden_units):
        hyper_string = ("HYPER-PARAMETERS"
                        "\n- hidden units: {4}"
                        "\n- batches: {0}"
                        "\n- epochs: {1}"
                        "\n- optimizer: {2}"
                        "\n- learning rate: {3}")
        return hyper_string.format(batch_size, epoch_count, optimizer, lr, hidden_units)

    def start_logging(self, batch_size, epoch_count, optimizer, lr, hidden_units):
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(str(self.log_file_path), 'a')
        self.__write_header(batch_size, epoch_count, optimizer, lr, hidden_units)
        self.__print_header(batch_size, epoch_count, optimizer, lr, hidden_units)
        self.start_time = datetime.now()

    def __write_header(self, batch_size, epoch_count, optimizer, lr, hidden_units):
        self.file.write("\nNN TRAINING LOG\n")
        self.file.write(datetime.now().strftime("%d.%m.%Y %H:%M:%S") + "\n")
        self.file.write("-------------------\n")
        self.file.write(self.__format_hyperparams(batch_size, epoch_count, type(optimizer), lr, hidden_units) + "\n\n")

    def __print_header(self, batch_size, epoch_count, optimizer, lr, hidden_units):
        print("\nBEGINNING TRAINING\n")
        print(self.__format_hyperparams(batch_size, epoch_count, type(optimizer), lr, hidden_units))
        print("")

    def __get_run_time(self):
        delta_t = datetime.now() - self.start_time
        str_delta_without_microsecs = str(delta_t).split('.')[0]
        return str_delta_without_microsecs

    def __print_epoch(self, epoch, train_loss, eval_loss, eval_accuracy):
        print(f"Epoch {epoch} [{self.__get_run_time()}]")
        print(f"Training loss: {train_loss:12.3f}")
        print(f"Evaluation loss: {eval_loss:10.3f}")
        print(f"Evaluation accuracy: {eval_accuracy * 100:1.3f}%")

    def __write_epoch(self, epoch, train_loss, eval_loss, eval_accuracy):
        epoch_string = (f"EPOCH {epoch} [{self.__get_run_time()}]"
                        f"\n> tr loss: {train_loss:0.3f}"
                        f"\n> ev loss: {eval_loss:0.3f}"
                        f"\n> ev accu: {eval_accuracy * 100:0.3f}%\n")
        self.file.write(epoch_string)

    def process_training_epoch(self, epoch, train_loss, eval_loss, eval_accuracy):
        self.__print_epoch(epoch, train_loss, eval_loss, eval_accuracy)
        self.__write_epoch(epoch, train_loss, eval_loss, eval_accuracy)

    def stop_logging(self):
        self.file.write(f"\nTotal run time: {self.__get_run_time()}\n")
        self.file.write(f"[x] END\n")
        self.file.close()
        print("\nTraining complete.\n")
