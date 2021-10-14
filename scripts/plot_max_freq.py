#!/usr/bin/env python

from os import makedirs, listdir
from os import path as osp
from PIL import Image as Image_

from cv_bridge import CvBridge
from sound_classification.msg import InSound
import message_filters
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import Spectrum

import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy import signal, interpolate

rospack = rospkg.RosPack()
save_dir = osp.join(rospack.get_path(
    "sound_classification"), "freq_data", "20211007_162340")
load_file = np.load(osp.join(save_dir, "max_freq.npy"))

time = load_file[0]
freq = load_file[1]

#print(freq)
#print(time)

save_dir2 = osp.join(rospack.get_path(
    "sound_classification"), "freq_data", "20211007_163712")
load_file2 = np.load(osp.join(save_dir2, "max_freq.npy"))

time2 = load_file2[0]
freq2 = load_file2[1]

fig, axs = plt.subplots()

axs.set_xlabel("x")
axs.set_ylabel("y")
axs.plot(time, freq, color="blue")
axs.plot(time2, freq2, color="red")
fig.tight_layout()
#plt.plot(time, freq)
#plt.plot(time2, freq2)


print(time)
print(time2)

#hokan
#t = np.linspace(4, 20, 1601)
t = np.linspace(4, 17, 1301)
f1 = interpolate.interp1d(time, freq)
y1 = f1(t)
axs.plot(t, y1, color="yellow")

#t2 = np.linspace(3, 17, 1401)
t2 = np.linspace(4, 17, 1301)
f2 = interpolate.interp1d(time2, freq2)
y2 = f2(t2)
axs.plot(t2, y2, color="purple")

plt.show()


distance, path = fastdtw(y1, y2, dist=euclidean)
#plt.plot(y1, label="y1")
#plt.plot(y2, label="y2")
#for x_, y_ in path:
#    plt.plot([x_, y_], [y1[x_], y2[y_]], color="gray", linestyle="dotted", linewidth=1)
#plt.legend()
#plt.show()

#print(path)

warp1 = []
warp2 = []
for i in range(len(path)):
    warp1.append(path[i][0])
    warp2.append(path[i][1])
plt.plot(warp1, warp2)
plt.legend()
plt.show()

for i in range(len(warp1)):
    print(warp1[i], warp2[i])
