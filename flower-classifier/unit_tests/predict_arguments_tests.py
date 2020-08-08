import unittest
from unittest.mock import patch

from predict_arguments import PredictArgumentParser


def get_predict_args(command_line: str):
    argv_list = command_line.split()
    with patch("sys.argv", argv_list):
        parser = PredictArgumentParser()
        args = parser.args
    return args


class RequiredArgumnetTests(unittest.TestCase):
    @staticmethod
    def test_input_and_checkpoint_are_recognized():
        get_predict_args(command_line="predict.py image.png model.pth")

    def test_two_positional_arguemnts_are_required(self):
        with self.assertRaises(SystemExit) as context:
            get_predict_args(command_line="predict.py model.pth")
            self.assertEqual(context.exception.code, 2)

    def test_input_is_correctly_stored(self):
        args = get_predict_args("predict.py path/to/image.png C:\\engineering\\my_model.pth")
        self.assertEqual(args.input, "path/to/image.png")

    def test_checkpoint_is_correctly_stored(self):
        args = get_predict_args("predict.py path/to/image.png C:\\engineering\\my_model.pth")
        self.assertEqual(args.checkpoint, "C:\\engineering\\my_model.pth")


class TopKTests(unittest.TestCase):
    @staticmethod
    def test_topk_is_recognized():
        get_predict_args("predict.py image.png checkpoint.pth --top_k 5")

    def test_topk_default_is_3(self):
        args = get_predict_args("predict.py image.png checkpoint.pth")
        self.assertEqual(args.top_k, 3)

    def test_topk_doesnt_accept_strings(self):
        with self.assertRaises(SystemExit) as context:
            get_predict_args("predict.py image.png checkpoint.pth --top_k=Powerslave")
            self.assertEqual(context.exception.code, 2)

    def test_topk_doesnt_accept_zero(self):
        with self.assertRaises(SystemExit) as context:
            get_predict_args("predict.py image.png checkpoint.pth --top_k=0")
            self.assertEqual(context.exception.code, 2)

    def test_topk_doesnt_accept_negative_ints(self):
        with self.assertRaises(SystemExit) as context:
            get_predict_args("predict.py image.png checkpoint.pth --top_k -1701")
            self.assertEqual(context.exception.code, 2)

    def test_topk_doesnt_accept_floats(self):
        with self.assertRaises(SystemExit) as context:
            get_predict_args("predict.py image.png checkpoint.pth --top_k 1.701")
            self.assertEqual(context.exception.code, 2)

    def test_topk_is_correctly_stored(self):
        args = get_predict_args("predict.py image.png checkpoint.pth --top_k 10")
        self.assertEqual(args.top_k, 10)


class CategoryNamesTests(unittest.TestCase):
    @staticmethod
    def test_category_names_is_recognized():
        get_predict_args("predict.py image.png checkpoint.pth --category_names cats.json")

    def test_default_category_names(self):
        default_file = "cat_to_name.json"
        args = get_predict_args("predict.py image.png checkpoint.pth")
        self.assertEqual(args.category_names, default_file)

    def test_category_names_is_correctly_stored(self):
        args = get_predict_args("predict.py image.png checkpoint.pth --category_names cats.json")
        self.assertEqual(args.category_names, "cats.json")


class GpuTests(unittest.TestCase):
    @staticmethod
    def test_gpu_is_recognized():
        get_predict_args("predict.py image.png checkpoint.pth --gpu")

    def test_default_gpu_is_false(self):
        args = get_predict_args("predict.py image.png checkpoint.pth")
        self.assertFalse(args.gpu)

    def test_gpu_doesnt_accept_any_values(self):
        with self.assertRaises(SystemExit) as context:
            get_predict_args("predict.py image.png checkpoint.pth --gpu=Maybe")
            self.assertEqual(context.exception.code, 2)

    def test_when_present_gpu_is_set_to_true(self):
        args = get_predict_args("predict.py image.png checkpoint.pth --gpu")
        self.assertTrue(args.gpu)


if __name__ == '__main__':
    unittest.main()
