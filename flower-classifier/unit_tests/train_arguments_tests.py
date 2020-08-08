import unittest
from unittest.mock import patch

from train_arguments import TrainArgumentParser


def get_train_args(command_line: str):
    argv_list = command_line.split()
    with patch("sys.argv", argv_list):
        parser = TrainArgumentParser()
        args = parser.args
    return args


class DataDirectoryParamTest(unittest.TestCase):
    def test_if_no_data_dir_given_uses_flowers_dir(self):
        args = get_train_args(command_line="train.py")
        self.assertEqual(args.data_dir, "flowers")

    def test_setting_data_dir_value(self):
        dir_arg = "Enterprise/engineering/"
        args = get_train_args(command_line=f"train.py {dir_arg}")
        self.assertEqual(args.data_dir, dir_arg)


class SaveDirParamTest(unittest.TestCase):
    @staticmethod
    def test_recognizes_save_dir_param():
        get_train_args(command_line="train.py data_dir --save_dir=save_there")

    def test_default_save_dir_is_current_dir(self):
        args = get_train_args(command_line="train.py")
        self.assertEqual(args.save_dir, "")

    def test_save_dir_takes_exactly_one_value(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args(command_line="train.py --save_dir")
            self.assertEqual(context.exception.code, 2)

    def test_setting_save_dir_value(self):
        save_dir_arg = "Voyager"
        args = get_train_args(command_line=f"train.py --save_dir {save_dir_arg}")
        self.assertEqual(args.save_dir, save_dir_arg)


class ArchParamTest(unittest.TestCase):
    @staticmethod
    def test_accepts_arch_param():
        get_train_args("train.py --arch=resnet50")

    def test_default_arch_is_resnet50(self):
        args = get_train_args("train.py")
        self.assertEqual(args.arch, "resnet50")

    def test_arch_takes_exactly_one_value(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --arch resnet50 vgg13 densenet121")
            self.assertEqual(context.exception.code, 2)

    def test_setting_arch_value(self):
        args = get_train_args("train.py --arch vgg13")
        self.assertEqual(args.arch, "vgg13")

    def test_doesnt_accept_arch_values_not_in_arch_dict(self):
        incorrect_arch = "VulcanNet"
        arch_dict_keys = TrainArgumentParser().arch_dict.keys()
        self.assertNotIn(incorrect_arch, arch_dict_keys)
        with self.assertRaises(SystemExit) as context:
            get_train_args(f"train.py --arch {incorrect_arch}")
            self.assertEqual(context.exception.code, 2)

    @staticmethod
    def test_accepts_every_arch_from_arch_dict():
        arch_dict_keys = TrainArgumentParser().arch_dict.keys()
        for arch_name in arch_dict_keys:
            get_train_args(command_line=f"train.py --arch {arch_name}")


class LearningRateParamTest(unittest.TestCase):
    @staticmethod
    def test_accepts_learning_rate_param():
        get_train_args("train.py --learning_rate 0.01")

    def test_default_learning_rate_is_0_0003(self):
        args = get_train_args("train.py")
        self.assertEqual(args.learning_rate, 0.0003)

    def test_learning_rate_doesnt_accept_strings(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --learning_rate three")
            self.assertEqual(context.exception.code, 2)

    def test_learning_rate_doesnt_accept_zero(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --learning_rate=0")
            self.assertEqual(context.exception.code, 2)

    def test_learning_rate_doesnt_accept_negative_floats(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --learning_rate -0.005")
            self.assertEqual(context.exception.code, 2)

    def test_setting_learning_rate_value(self):
        value = 1.701
        args = get_train_args(f"train.py --learning_rate={value}")
        self.assertEqual(args.learning_rate, value)


class HiddenUnitsParamTest(unittest.TestCase):
    @staticmethod
    def test_accepts_hidden_units_param():
        get_train_args("train.py --hidden_units=1024")

    def test_default_hidden_units_is_1536(self):
        args = get_train_args("train.py")
        self.assertEqual(args.hidden_units, 1536)

    def test_hidden_units_doesnt_accept_strings(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --hidden_units one_billion_and_ten")
            self.assertEqual(context.exception.code, 2)

    def test_hidden_units_doesnt_accept_floats(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --hidden_units 1024.5")
            self.assertEqual(context.exception.code, 2)

    def test_hidden_units_doesnt_accept_zero(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --hidden_units 0")
            self.assertEqual(context.exception.code, 2)

    def test_hidden_units_doesnt_accept_negative_ints(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --hidden_units -512")
            self.assertEqual(context.exception.code, 2)

    def test_setting_hidden_units_value(self):
        value = 1701
        args = get_train_args(f"train.py --hidden_units={value}")
        self.assertEqual(args.hidden_units, value)


class EpochsParamTest(unittest.TestCase):
    @staticmethod
    def test_accepts_epochs_param():
        get_train_args("train.py --epochs 10")

    def test_default_epochs_is_5(self):
        args = get_train_args("train.py")
        self.assertEqual(args.epochs, 5)

    def test_epochs_doesnt_accept_strings(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --epochs=ten")
            self.assertEqual(context.exception.code, 2)

    def test_epochs_doesnt_accept_floats(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --epochs=10.5")
            self.assertEqual(context.exception.code, 2)

    def test_epochs_doesnt_accept_negative_integers(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --epochs=-8")
            self.assertEqual(context.exception.code, 2)

    def test_seting_epochs_value(self):
        value = 20
        args = get_train_args(f"train.py --epochs={value}")
        self.assertEqual(args.epochs, value)


class GpuParamTests(unittest.TestCase):
    @staticmethod
    def test_accepts_gpu_param():
        get_train_args("train.py --gpu")

    def test_when_not_given_gpu_is_set_to_false(self):
        args = get_train_args("train.py")
        self.assertFalse(args.gpu)

    def test_gpu_doesnt_take_values(self):
        with self.assertRaises(SystemExit) as context:
            get_train_args("train.py --gpu=True")
            self.assertEqual(context.exception.code, 2)

    def test_when_given_gpu_is_set_to_true(self):
        args = get_train_args("train --gpu")
        self.assertTrue(args.gpu)


if __name__ == '__main__':
    unittest.main()
