#!/usr/bin/env python

from PIL import Image as Image_
import os
import numpy as np

path = "/home/nakaotatsuya/audio_ws/src/sound_classification/train_data/dataset"
img = Image_.open(os.path.join(path, "test_door00003.png"))
x_train = np.array(img)
print(len(x_train[0][0]))

img = img.resize((32,32))
x_test = np.array(img)
print(len(x_test[0][0]))
img.save("test.png")
