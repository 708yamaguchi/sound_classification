#!/usr/bin/env python

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision

import matplotlib
import numpy as np
from os import makedirs
import os.path as osp
from PIL import Image as Image_
import rospkg
import rospy

from nin.nin import NIN
from lstm.lstm import LSTM

matplotlib.use("Agg")

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, path=None, random=True, train_data="train_data"):
        rospack = rospkg.RosPack()
        self.root = osp.join(rospack.get_path(
            "sound_classification"), train_data)
        if path is not None:
            self.base = torchvision.datasets.ImageFolder(
                root=osp.join(self.root, path))
        self.random = random
        # how many classes to be classified
        self.n_class = 0
        self.target_classes = []
        with open(osp.join(self.root, 'n_class.txt'), mode='r') as f:
            for row in f:
                self.n_class += 1
                self.target_classes.append(row)
        # Load mean image of dataset
        mean_img_path = osp.join(rospack.get_path('sound_classification'),
                                 train_data, 'dataset', 'mean_of_dataset.png')
        mean = np.array(Image_.open(mean_img_path), np.float32).transpose(
            (2, 0, 1))  # (height, width, channel) -> (channel ,height, width), rgb
        #self.mean = mean.astype(chainer.get_dtype())
        self.mean = mean

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, label = self.base.samples[i]
        image = np.array(Image_.open(base.samples[0][0]), dtype=np.float32).transpose((2,0,1))
        label = np.array(label, dtype=np.int32)
        image = self.process_image(image)

        image = torch.Tensor(image, dtype=torch.float)
        label = torch.Tensor(label, dtype=torch.float)
        return image, label

    def process_image(self, image):
        ret = image - self.mean
        ret *= (1.0 / 255.0)
        return ret

def load_model(model_name, n_class):
    archs = {
        #'nin': NIN,
        #'vgg16': VGG16BatchNormalization
        "lstm": LSTM
    }
    model = archs[model_name](n_class=n_class)
    return model

def main():
    rospack = rospkg.RosPack()

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument("-t", "--train_data", type=str, default="train_data")
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-m', '--model', type=str,
                        choices=["lstm"], default="lstm",
                        help='Neural network model to use dataset')
    args = parser.parse_args()


    if args.gpu == -1:
        device = "cpu"
    elif args.gpu == 0:
        device = "cuda"
    #device = chainer.cuda.get_device_from_id(args.gpu)  # for python2
    batchsize = 32
    # Path to training image-label list file
    #train_labels = osp.join(rospack.get_path('sound_classification'),
    #                        args.train_data, 'dataset', 'train_images.txt')
    # Path to validation image-label list file
    #val_labels = osp.join(rospack.get_path('sound_classification'),
    #                      args.train_data, 'dataset', 'test_images.txt')

    # Initialize the model to train
    print('Device: {}'.format(device))
    print('Model: {}'.format(args.model))
    #print('Dtype: {}'.format(chainer.config.dtype))
    print('Minibatch-size: {}'.format(batchsize))
    print('epoch: {}'.format(args.epoch))
    print('')

    train = PreprocessedDataset(path="dataset_torch", train_data=args.train_data)
    val = PreprocessedDataset(path="dataset_torch_val", val_labels, False, train_data=args.train_data)

    model = load_model(args.model, train.n_class)
    model = model.to(device)

    trainloader = torch.utils.data.DataLoader(train, batch_size=batchsize, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(val, batch_size=batchsize, shuffle=False, num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    out = osp.join(rospack.get_path("sound_classification"),
                   args.train_data, "result_torch", args.model)
    if not osp.exists(out):
        makedirs(out)

