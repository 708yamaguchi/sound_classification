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
import os
import datetime as dt

class MaxFreq(object):
    def __init__(self):
        rospack = rospkg.RosPack()
        self.time_list = []
        self.max_freq_list = []
        self.start = rospy.Time.now()
        self.flag = True
        dt_now = dt.datetime.now()
        yyyymmdd = dt_now.strftime("%Y%m%d_%H%M%S")
        self.save_dir = osp.join(rospack.get_path(
            "sound_classification"), "freq_data", yyyymmdd)

        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)

        in_sound_sub = message_filters.Subscriber("~in_sound", InSound)
        freq_sub = message_filters.Subscriber("~freq", Spectrum)
        subs = [in_sound_sub, freq_sub]
        ts = message_filters.TimeSynchronizer(subs, 100000)
        ts.registerCallback(self._cb)


    def _cb(self, *args):
        in_sound = args[0].in_sound
        amplitude = np.array(args[1].amplitude)
        frequency = np.array(args[1].frequency)

        #print(frequency.shape)
        low_cut_amplitude = amplitude[400:1800]
        low_cut_frequency = frequency[400:1800]
        print(low_cut_frequency)
        #print(low_cut_frequency[0])

        time = (args[1].header.stamp - self.start).to_sec()
        if in_sound:
            max_freq = low_cut_frequency[np.argmax(low_cut_amplitude)]
            #print(max_freq)
            rospy.loginfo(time)
            rospy.loginfo(max_freq)
            if max_freq < 5000 and max_freq > 3500:
                self.time_list.append(time)
                self.max_freq_list.append(max_freq)
            
        if time >= 25.0 and self.flag:
            t_f = np.array([self.time_list, self.max_freq_list])
            np.save(osp.join(self.save_dir, "max_freq.npy"), t_f)
            self.flag = False

if __name__ == "__main__":
    rospy.init_node("max_freq")
    a = MaxFreq()
    rospy.spin()
