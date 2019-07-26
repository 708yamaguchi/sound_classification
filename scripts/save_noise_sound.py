#!/usr/bin/env python

# This node saves noise sound (environment sound)

import numpy as np
import os.path as osp
import rospkg
import rospy
from sound_classification.msg import Spectrum
import time


if __name__ == '__main__':
    rospy.init_node('save_noise_sound.py', anonymous=True)
    rospy.sleep(0.1)  # do not save typing sound
    time_start = time.time()
    mean_noise_sound = None
    sound_count = 0
    while(time.time() - time_start < 3.0):
        msg = rospy.wait_for_message('/microphone/spectrum', Spectrum)
        if mean_noise_sound is None:
            mean_noise_sound = np.array(msg.spectrum)
        else:
            mean_noise_sound = mean_noise_sound + np.array(msg.spectrum)
        sound_count += 1
    # mean noise sound
    mean_noise_sound = mean_noise_sound / sound_count
    # save noise sound
    rospack = rospkg.RosPack()
    file_name = osp.join(rospack.get_path(
        'sound_classification'), 'scripts', 'mean_noise_sound')
    np.save(file_name, mean_noise_sound)
    rospy.loginfo('saved {}'.format(file_name))
    rospy.signal_shutdown('finish')
