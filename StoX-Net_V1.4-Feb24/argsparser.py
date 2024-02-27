import argparse
import resnet


def get_parser():
    model_names = sorted(name for name in resnet.__dict__
                         if name.islower() and not name.startswith("__")
                         and name.startswith("resnet")
                         and callable(resnet.__dict__[name]))

    parser = argparse.ArgumentParser(description='Property ResNets for CIFAR10 in pytorch')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_1w1a',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: resnet32)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                        type=bool, help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='saved_models', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    parser.add_argument('--time-steps', default=4, type=int, metavar='N',
                        help='Maximum time steps for each MTJ (default: 1)')
    parser.add_argument('--num-ab', default=4, type=int, metavar='N',
                        help='Denotes maximum number of activation bits (default: 1)')
    parser.add_argument('--num-wb', default=4, type=int, metavar='N',
                        help='Number of weight bits (default: 1)')
    parser.add_argument('--ab-sw', default=4, type=int, metavar='N',
                        help='Number of activation bits per stream (default: 1)')
    parser.add_argument('--wb-sw', default=4, type=int, metavar='N',
                        help='Number of weight bits per slices (default: 1)')
    parser.add_argument('--subarray-size', default=256, type=int, metavar='N',
                        help='Maximum subarray size for partial sums')
    parser.add_argument('--dataset', dest='dataset', help='Choose a dataset to run the network on from'
                        '{MNIST, CIFAR10, CIFAR100, tiny_imagenet}', default='MNIST', type=str)
    parser.add_argument('--input-pos-only', default=False, type=bool,
                        help='Choose whether forward pass activations are positive only')

    return parser
