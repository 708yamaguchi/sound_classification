#!/usr/bin/env python

import math
from sensor_msgs.msg import Image
from std_msgs.msg import String
import rospy
import rospkg
import numpy as np

class MakeUnknown:
    def __init__(self):
        rospy.init_node("make_unknown")
        self.sub = rospy.Subscriber(
            "/object_class_by_image", String, self.cb, queue_size=1)

    def 
