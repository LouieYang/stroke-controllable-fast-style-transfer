import argparse
import os
import shutil
import tensorflow as tf
from model import Model
from utils import mkdir_if_not_exists

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def build_parser():
    parser = argparse.ArgumentParser()

    # Input Options
    parser.add_argument('--style', type=str, dest='style', help='style image path',
                        default='./examples/style/01.png')

    parser.add_argument('--batch_size', type=int, dest='batch_size', help='batch size', 
                        default=2)
    parser.add_argument('--max_iter', type=int, dest='max_iter', help='max iterations', 
                        default=2e4)

    parser.add_argument('--learning_rate', type=float, dest='learning_rate', 
                        default=1e-3)
    parser.add_argument('--iter_print', type=int, dest='iter_print', default=5e2)

    parser.add_argument('--checkpoint_iterations', type=int, dest='checkpoint_iterations',
                        help='checkpoint frequency', default=1e3)
    parser.add_argument('--train_path', type=str, dest='train_path',
                        help='path to training content images folder', default="./data/MSCOCO")

    # Weight Options
    parser.add_argument('--content_weight', type=float, dest="content_weight",
                        help='content weight (default %(default)s)', default=80)
    parser.add_argument('--style_weight', type=float, dest="style_weight",
                        help='style weight (default %(default)s)', default=1e2)
    parser.add_argument('--tv_weight', type=float, dest="tv_weight",
                        help="total variation regularization weight (default %(default)s)",
                        default=2e2)

    # Finetune Options
    parser.add_argument('--continue_train', type=bool, dest='continue_train', default=False)

    # Others
    parser.add_argument('--sample_path', type=str, dest="sample_path", 
                        default='./examples/content/01.jpg')

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    train_model = Model(sess, args)

    style_image_basename = os.path.basename(args.style)
    style_image_basename = style_image_basename[:style_image_basename.find(".")]

    args.checkpoint_dir = os.path.join("./examples/checkpoint", style_image_basename)
    args.serial = os.path.join("./examples/serial", style_image_basename)

    print("[*] Checkpoint Directory: {}".format(args.checkpoint_dir))
    print("[*] Serial Directory: {}".format(args.serial))
    mkdir_if_not_exists(args.serial, args.checkpoint_dir)

    if args.continue_train:
        train_model.finetune_model(args)
    else:
        train_model.train(args)

if __name__ == "__main__":
    main()
