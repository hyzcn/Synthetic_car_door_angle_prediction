import argparse
import os.path as osp
import ast
class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="normal netowork.")
        parser.add_argument("--command", type=str, default='train',help="available mode: train or test.")
        parser.add_argument("--model-name", type=str, default='resnet',help="backbone for the network.")
        parser.add_argument("--part-name", type=str, default='all',help="which part to predict, available: all, fl.")
        parser.add_argument("--batch-size", type=int, default=1, help="input batch size.")
        parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs.")
        parser.add_argument("--num-images", type=int, default=97200, help="total number of images in train set")
        parser.add_argument("--train-num", type=int, default=45000, help="number of images in train set")
        parser.add_argument("--test-num", type=int, default=9720, help="number of images in test set")
        parser.add_argument("--feature-extract", type=ast.literal_eval, default=False, help="fine-tune or fix the feature while transfer learning.")
        parser.add_argument("--data-range", type=int, default=60, help="max range of prediction factors.")
        parser.add_argument("--data-dir", type=str, default='datasets/train/preset_car_data/', help="Path to the directory containing the train images.")
        parser.add_argument("--test-dir", type=str, default='datasets/all_test/preset_all_random/', help="Path to the directory containing the test images.")
        parser.add_argument("--train-name-dir", type=str, default='ImageSets/preset_texture_all_train.txt', help="Path to the file containing training names.")
        parser.add_argument("--test-name-dir", type=str, default='ImageSets/preset_texture_all_test.txt', help="Path to the file containing testing names.")
        parser.add_argument("--model-dir", type=str, default='params/{}_ft_{}_norm.pkl', help="Path to the directory saving the model.")
        parser.add_argument("--plot-dir", type=str, default='plots/{}_ft_{}.jpg', help="Path to the directory saving the plots.")
        parser.add_argument("--output-dir", type=str, default='outputs/{}_ft_{}.txt', help="Path to the directory saving the output text.")
        parser.add_argument("--html-dir", type=str, default='htmls/{}_ft_{}.txt', help="Path to the directory saving html texts.")
        parser.add_argument("--train-gt-dir", type=str, default='gt_dict/preset_car_all_gt.npy', help="Path to test set gt.")
        parser.add_argument("--test-gt-dir", type=str, default='gt_dict/preset_car_all_gt.npy', help="Path to test set gt.")
        parser.add_argument("--test-spatial", type=ast.literal_eval, default=False, help="whether test the model on spatial interferenced dataset.")

        return parser.parse_args()

    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # # save to the disk
        # file_name = osp.join(args.snapshot_dir, 'opt.txt')
        # with open(file_name, 'wt') as args_file:
        #     args_file.write(message)
        #     args_file.write('\n')

