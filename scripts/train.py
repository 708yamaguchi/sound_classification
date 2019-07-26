#!/usr/bin/env python

# mainly copied from chainer/train_imagenet.py
# https://github.com/chainer/chainer/blob/master/examples/imagenet/train_imagenet.py

import argparse

import chainer
from chainer import dataset
from chainer import training
from chainer.training import extensions
from chainer_modules import dali_util
from chainer_modules import nin

import matplotlib
import numpy as np
import os.path as osp
import rospkg

matplotlib.use('Agg')  # necessary not to raise Tcl_AsyncDelete Error


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype(chainer.get_dtype())
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        image = image[[2, 1, 0], :, :]  # bgr -> rgb

        # image -= self.mean[:, top:bottom, left:right]
        image -= self.mean  # image is np.astype(np.float32)
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


def main():
    archs = {  # only NIN is available now
        # 'alex': alex.Alex,
        # 'googlenet': googlenet.GoogLeNet,
        # 'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
        # 'resnet50': resnet50.ResNet50,
        # 'resnext50': resnext50.ResNeXt50,
    }

    dtypes = {
        'float16': np.float16,
        'float32': np.float32,
        'float64': np.float64,
    }
    rospack = rospkg.RosPack()

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('--train', default=osp.join(
        rospack.get_path('sound_classification'),
        'train_data', 'dataset', 'train_images.txt'),
                        help='Path to training image-label list file')
    parser.add_argument('--val', default=osp.join(
        rospack.get_path('sound_classification'),
        'train_data', 'dataset', 'test_images.txt'),
                        help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--dtype', choices=dtypes, help='Specify the dtype '
                        'used. If not supplied, the default dtype is used')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default=osp.join(
        rospack.get_path('sound_classification'),
        'scripts', 'chainer_modules', 'mean.npy'),
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default=osp.join(
        rospack.get_path('sound_classification'),
        'scripts', 'result'),
                        help='Output directory')
    parser.add_argument('--root', '-R', default=osp.join(rospack.get_path(
        'sound_classification'), 'train_data', 'dataset'),
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--dali', action='store_true')
    parser.set_defaults(dali=False)
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # device = chainer.get_device(args.device)  # for python3
    device = chainer.cuda.get_device(args.device)  # for python2

    # Set the dtype if supplied.
    if args.dtype is not None:
        chainer.config.dtype = args.dtype

    print('Device: {}'.format(device))
    print('Dtype: {}'.format(chainer.config.dtype))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Initialize the model to train
    n_class = 0
    with open(osp.join(args.root, 'n_class.txt'), mode='r') as f:
        for row in f:
            n_class += 1
    model = archs[args.arch](n_class=n_class)
    if args.initmodel:
        print('Load model from {}'.format(args.initmodel))
        chainer.serializers.load_npz(args.initmodel, model)
    model.to_device(device)
    device.use()

    # Load the mean file
    mean = np.load(args.mean)
    if args.dali:
        if not dali_util._dali_available:
            raise RuntimeError('DALI seems not available on your system.')
        if device.xp is not chainer.backend.cuda.cupy:
            raise RuntimeError('Using DALI requires GPU device. Please '
                               'specify it with --device option.')
        num_threads = args.loaderjob
        if num_threads is None or num_threads <= 0:
            num_threads = 1
        ch_mean = list(np.average(mean, axis=(1, 2)))
        ch_std = [255.0, 255.0, 255.0]
        # Setup DALI pipelines
        train_pipe = dali_util.DaliPipelineTrain(
            args.train, args.root, model.insize, args.batchsize,
            num_threads, device.device.id, True, mean=ch_mean, std=ch_std)
        val_pipe = dali_util.DaliPipelineVal(
            args.val, args.root, model.insize, args.val_batchsize,
            num_threads, device.device.id, False, mean=ch_mean, std=ch_std)
        train_iter = chainer.iterators.DaliIterator(train_pipe)
        val_iter = chainer.iterators.DaliIterator(val_pipe, repeat=False)
        # converter = dali_converter
        converter = dali_util.DaliConverter(mean=mean, crop_size=model.insize)
    else:
        # Load the dataset files
        train = PreprocessedDataset(args.train, args.root, mean, model.insize)
        val = PreprocessedDataset(args.val, args.root, mean, model.insize,
                                  False)
        # These iterators load the images with subprocesses running in parallel
        # to the training/validation.
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=args.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(
            val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)
        converter = dataset.concat_examples

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = 100, 'iteration'
    log_interval = 100, 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, converter=converter,
                                        device=device), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        target=model, filename='model_best.npz'),
        trigger=chainer.training.triggers.MinValueTrigger(
            key='validation/main/loss',
            trigger=val_interval))
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='iteration', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='iteration', file_name='accuracy.png'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
