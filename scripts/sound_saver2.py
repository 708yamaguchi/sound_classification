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
from jsk_recognition_msgs.msg import Accuracy

class SoundSaver(object):
    """
    Collect spectrogram with sound class, only when the robot is in sound.
    if save_when_sound is False, you can save spectrograms during no sound.
    """

    def __init__(self):
        # Config for saving spectrogram
        self.target_class = rospy.get_param('~target_class')
        self.save_raw_spectrogram = rospy.get_param('~save_raw_spectrogram')
        rospack = rospkg.RosPack()
        train_data = rospy.get_param("~train_data")
        self.train_dir = osp.join(rospack.get_path(
            'sound_classification'), train_data)
        if not osp.exists(self.train_dir):
            makedirs(self.train_dir)
        self.image_save_dir = osp.join(
            self.train_dir, 'original_spectrogram', self.target_class)
        if not osp.exists(self.image_save_dir):
            makedirs(self.image_save_dir)
            
        self.raw_image_save_dir = osp.join(self.image_save_dir, 'raw')
        self.param_save_dir = osp.join(self.image_save_dir, "param")
        if not osp.exists(self.raw_image_save_dir):
            makedirs(self.raw_image_save_dir)
        if not osp.exists(self.param_save_dir):
            makedirs(self.param_save_dir)
        noise = np.load(osp.join(self.train_dir, 'noise.npy'))
        np.save(osp.join(self.image_save_dir, 'noise.npy'), noise)
        # ROS
        self.bridge = CvBridge()
        self.save_data_rate = rospy.get_param('~save_data_rate')
        self.save_when_sound = rospy.get_param('~save_when_sound')
        self.in_sound = False
        self.spectrogram_msg = None
        self.spectrogram_raw_msg = None
        in_sound_sub = message_filters.Subscriber('~in_sound', InSound)
        img_sub = message_filters.Subscriber('~input', Image)
        img_raw_sub = message_filters.Subscriber('~input_raw', Image)
        #param_sub = message_filters.Subscriber("skeleten_test/output_param", Accuracy)
        #subs = [in_sound_sub, img_sub, img_raw_sub, param_sub]
        subs = [in_sound_sub, img_sub]
        if self.save_raw_spectrogram:
            subs.append(img_raw_sub)
        ts = message_filters.TimeSynchronizer(subs, 100000)
        #ts = message_filters.ApproximateTimeSynchronizer(subs, 100000, slop=0.1)
        ts.registerCallback(self._cb)

        param_sub = rospy.Subscriber("skeleten_test/output_param", Accuracy, self.callback)
        self.param = None
        rospy.Timer(rospy.Duration(1. / self.save_data_rate), self.timer_cb)

    def callback(self, msg):
        self.param = np.array([msg.accuracy])
        
    def _cb(self, *args):
        in_sound = args[0].in_sound
        # rospy.logerr('in_sound: {}'.format(in_sound))
        if self.save_when_sound is False:
            in_sound = True
        if in_sound:
            self.spectrogram_msg = args[1]
            if self.save_raw_spectrogram:
                self.spectrogram_raw_msg = args[2]
            #print(args[3])
            #self.param = np.array([args[3].accuracy])
        else:
            self.spectrogram_msg = None
            if self.save_raw_spectrogram:
                self.spectrogram_raw_msg = None
            #self.param = np.array([0])

    def timer_cb(self, timer):
        """
        Main process of NoiseSaver class
        Save spectrogram data at self.save_data_rate
        """

        if self.spectrogram_msg is None or self.spectrogram_raw_msg is None:
            return
        if self.param is None:
            return
        else:
            file_num = len(
                listdir(self.image_save_dir)) + 1  # start from 00001.npy
            file_name = osp.join(
                self.image_save_dir, '{}_{:0=5d}.png'.format(
                    self.target_class, file_num))
            mono_spectrogram = self.bridge.imgmsg_to_cv2(self.spectrogram_msg)
            Image_.fromarray(mono_spectrogram).save(file_name)
            # self.spectrogram_msg = None
            rospy.loginfo('save spectrogram: ' + file_name)

            param_file_name = osp.join(
                self.param_save_dir, "{}_{:0=5d}.txt".format(
                    self.target_class, file_num))
            np.savetxt(param_file_name, self.param)

            #rospy.loginfo("param: " + self.param)
            if self.save_raw_spectrogram:
                file_name_raw = osp.join(
                    self.raw_image_save_dir, '{}_{:0=5d}_raw.png'.format(
                        self.target_class, file_num))
                try:
                    mono_spectrogram_raw = self.bridge.imgmsg_to_cv2(
                        self.spectrogram_raw_msg, desired_encoding='32FC1')
                except AttributeError:
                    return
                _max = mono_spectrogram_raw.max()
                _min = mono_spectrogram_raw.min()
                mono_spectrogram_raw = (mono_spectrogram_raw - _min) / (_max - _min) * 255.0
                mono_spectrogram_raw = mono_spectrogram_raw.astype(np.uint8)
                Image_.fromarray(mono_spectrogram_raw).save(file_name_raw)
                # self.spectrogram_raw_msg = None
                rospy.loginfo('save spectrogram: ' + file_name_raw)


if __name__ == '__main__':
    rospy.init_node('sound_saver')
    a = SoundSaver()
    rospy.spin()
