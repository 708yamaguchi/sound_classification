#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import rospkg
import numpy as np
import message_filters
import sympy
import tf
import tf.transformations
from os import makedirs, listdir
from os import path as osp

#from tmc_eus_py.coordinates import Coordinates
from geometry_msgs.msg import Point, PointStamped, Point, Quaternion, PoseArray
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import PeoplePoseArray, Accuracy
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from sound_classification.msg import InSound

def find_pose(limb, msg):
    if isinstance(limb, list):
        for l in limb:
            p = find_pose(l, msg)
            if p is not None:
                return p
        return None
    try:
        idx = msg.limb_names.index(limb)
        return msg.poses[idx]
    except ValueError:
        return None
    
class SkeletenTest():
    def __init__(self):
        #sub = rospy.Subscriber("/people_pose_estimation_2d/pose", PeoplePoseArray, self.callback, queue_size=1)

        self.flag = True
        sub_pose = message_filters.Subscriber("/people_pose_estimation_2d/pose", PeoplePoseArray)
        sub_in_sound = message_filters.Subscriber("/sound_detector_volume/in_sound", InSound)
        subs = [sub_pose, sub_in_sound]
        ts = message_filters.ApproximateTimeSynchronizer(subs, 100, slop=0.1)
        ts.registerCallback(self.callback)
        self.pub = rospy.Publisher("~output", PoseArray, queue_size=1)
        self.pub_marker = rospy.Publisher("~output_marker", Marker, queue_size=1)
        self.pub_param = rospy.Publisher("~output_param", Accuracy, queue_size=1)
        self.start_time = 0
        
    def callback(self, msg, msg_in_sound):
        #print("aaa")
        if msg_in_sound.in_sound and self.flag:
            self.start_time = rospy.Time.now()
            print("a")
            self.flag = False
        if len(msg.poses) == 0:
            return
        
        person_pose = msg.poses[0]

        #lfinger0 = find_pose(["LHand0"], person_pose)
        
        #lfinger1 = find_pose(["LHand1"], person_pose)
        #lfinger2 = find_pose(["LHand2"], person_pose)
        #lfinger3 = find_pose(["LHand3"], person_pose)
        #lfinger4 = find_pose(["LHand4"], person_pose)
        
        lfinger5 = find_pose(["LHand5"], person_pose)
        #lfinger6 = find_pose(["LHand6"], person_pose)
        #lfinger7 = find_pose(["LHand7"], person_pose)
        #lfinger8 = find_pose(["LHand8"], person_pose)
        
        lfinger9 = find_pose(["LHand9"], person_pose)
        lfinger13 = find_pose(["LHand13"], person_pose)
        lfinger17 = find_pose(["LHand17"], person_pose)
        
        pub_msg = PoseArray()
        pub_msg.header = msg.header

        pub_marker_msg = Marker()
        pub_marker_msg.header = msg.header
        pub_marker_msg.header.frame_id = "head_rgbd_sensor_rgb_frame"
        pub_marker_msg.type = 5
        pub_marker_msg.color.r = 1.0
        pub_marker_msg.color.a = 1.0

        pub_marker_msg.scale.x = 0.01
        pub_marker_msg.scale.y = 0.01
        pub_marker_msg.scale.z = 0.01

        pub_param = Accuracy()
        pub_param.header = msg.header

        #if lfinger5:
        #    pub_msg.poses.append(lfinger5)
        # if lfinger6:
        #     pub_msg.poses.append(lfinger6)
        # if lfinger7:
        #     pub_msg.poses.append(lfinger7)
        # if lfinger8:
        #     pub_msg.poses.append(lfinger8)
        #if lfinger9:
        #    pub_msg.poses.append(lfinger9)

        #if lfinger13:
        #    pub_msg.poses.append(lfinger13)
        #if lfinger17:
        #    pub_msg.poses.append(lfinger17)

        # if lfinger0:
        #     pub_msg.poses.append(lfinger0)
        # if lfinger1:
        #     pub_msg.poses.append(lfinger1)
        # if lfinger2:
        #     pub_msg.poses.append(lfinger2)
        # if lfinger3:
        #     pub_msg.poses.append(lfinger3)
        # if lfinger4:
        #     pub_msg.poses.append(lfinger4)

        if lfinger5 and lfinger9 and lfinger13 and lfinger17:
            pub_msg.poses.append(lfinger5)
            pub_msg.poses.append(lfinger9)
            pub_msg.poses.append(lfinger13)
            pub_msg.poses.append(lfinger17)
            
            pub_marker_msg.points = []
            pub_marker_msg.points.append(pub_msg.poses[0].position)
            pub_marker_msg.colors.append(ColorRGBA(r=1.0, a=1.0))
            pub_marker_msg.points.append(pub_msg.poses[3].position)
            pub_marker_msg.colors.append(ColorRGBA(r=1.0, a=1.0))

            mother = pub_msg.poses[0].position
            child = pub_msg.poses[1].position

            norm_vector = np.array([0, -1, 0])
            child_to_mother_vector = np.array([mother.x - child.x, mother.y - child.y, mother.z - child.z])
            finger_angle = np.dot(norm_vector, child_to_mother_vector) / np.sqrt(np.dot(child_to_mother_vector, child_to_mother_vector))
            #print(a)

            #正規化
            finger_angle = (finger_angle - (-0.5)) / (1.0 - (-0.5))

            if self.start_time:
                elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
                #print(elapsed_time)
                elapsed_time /= 30.0

                param = elapsed_time * finger_angle
                print(param)
                pub_param.accuracy = param

        self.pub.publish(pub_msg)
        self.pub_marker.publish(pub_marker_msg)
        self.pub_param.publish(pub_param)

if __name__ == "__main__":
    rospy.init_node("skeleten_test")
    a = SkeletenTest()
    rospy.spin()
