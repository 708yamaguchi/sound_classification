#!/usr/bin/env python

from cv_bridge import CvBridge
import os
import os.path as osp
from PIL import Image as Image_
import rospkg
import rospy
from sensor_msgs.msg import Image

class SaveSpectrogram3:
    def __init__(self):
        rospy.init_node("save_spectrogram_3", anonymous=True)
