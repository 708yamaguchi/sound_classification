1#!/usr/bin/env python

import numpy as np
from sklearn import svm
from PIL import Image as Image_
import os

class OneClassSvm():
    def __init__(self):
        self.x_test = []
        self.x_train = []

    def train_data(self):
        path = "/home/nakaotatsuya/audio_ws/src/sound_classification/train_data/dataset"
        count = 0
        for curDir, dirs, files in os.walk(path):
            for file in files:
                count += 1
                if file.endswith(".png") and count<1000:
                    print(file)
                    img = Image_.open(os.path.join(path, file))
                    img = img.resize((16,16))
                    self.x_train.append(np.array(img).flatten())
                    #print(x_train.shape)
        self.x_train = np.array(self.x_train)
        #print(type(self.x_train))
        #print(len(x_train[0][0][0]))
        return self.x_train

    def test_data(self):
        path = "/home/nakaotatsuya/audio_ws/src/sound_classification/train_data/dataset"
        count = 0
        for curDir, dirs, files in os.walk(path):
            for file in files:
                count += 1
                if file.endswith(".png") and count > 990 and count < 1010:
                    #pass
                    print(file)
                    img = Image_.open(os.path.join(path, file))
                    img = img.resize((16,16))
                    self.x_test.append(np.array(img).flatten())
                    #print(x_train.shape)
        self.x_test = np.array(self.x_test)
        #print(type(self.x_test))
        #print(len(x_train[0][0][0]))
        return self.x_test

    #def execute_OCSVM(self):
    #    clf = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma=0.001)
    #    print("ccc")
    #    clf.fit(self.x_train)
    #    print("ddd")
    #    pred = clf.predict(self.x_test)
    #    return pred

    #def evaluate(self, pred):
    #    return 0

if __name__ == "__main__":
    ocs = OneClassSvm()
    x_train = ocs.train_data()
    print("---------------")
    x_test = ocs.test_data()
    print("aaa")
    clf = svm.OneClassSVM(nu=0.25, kernel="rbf", gamma="auto")
    clf.fit(x_train)
    print("bbb")
    pred = clf.predict(x_test)
    print(pred)
    
    #evaluate(pred)
    
    #im = np.array(Image_.open("/home/nakaotatsuya/audio_ws/src/sound_classification/train_data/dataset/mean_of_dataset.png"))
    #print(im.dtype)
    #print(im.ndim)
    #print(im.shape)
    #x_train = np.empty((256,256,3))

    # x_train = []
    # path = "/home/nakaotatsuya/audio_ws/src/sound_classification/train_data/dataset"
    # for curDir, dirs, files in os.walk(path):
    #     for file in files:
    #         if file.endswith(".png"):
    #             #pass
    #             print(file)
    #             #np.array(Image_.open(os.path.join(path,file))).flatten
    #             x_train.append(np.array(Image_.open(os.path.join(path, file))).flatten())
    #             #print(x_train)

    # x_train = np.array(x_train)
    # print(type(x_train))
    # #print(len(x_train[0]))
