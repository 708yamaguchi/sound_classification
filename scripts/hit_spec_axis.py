#!/usr/bin/env python

from PIL import Image as Image_
import matplotlib.pyplot as plt
import numpy as np

import os
import os.path as osp
import rospkg
import rospy

from matplotlib.ticker import MaxNLocator
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CutSpectrogram:
    def __init__(self):
        rospy.init_node("cut_spectrogram", anonymous=True)
        train_data = rospy.get_param("~train_data", "train_data")

        rospack = rospkg.RosPack()
        self.target_class = rospy.get_param(
            '~target_class', 'unspecified_data')
        self.save_dir = osp.join(rospack.get_path(
            "sound_classification"), train_data)
        self.image_save_dir = osp.join(
            self.save_dir, "cut_spectrogram", self.target_class)
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)
        self.bridge=CvBridge()
        #publisher
        self.cut_spectrogram_pub = rospy.Publisher(
            "~cut_spectrogram", Image, queue_size=10)
        #subscriber
        hit_spec_sub = rospy.Subscriber(
            "~hit_spectrogram", Image, self.cb, queue_size=10)
        self.img_msg= Image()

    def cb(self, msg):
        hit_spectrogram = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        a = hit_spectrogram.transpose(1,0,2)
        print(a.shape)
        
        distance =[]
        for i in range(a.shape[0]):
            dist = 0
            for j in range(a.shape[1]):
                dist += (a[i][j][0] - 0)**2 + (a[i][j][1] - 0)**2 + (a[i][j][2] - 127) ** 2
            distance.append(dist)

        print(distance)
        #print(len(distance))

        sequence = 3
        first = True
        for i in range(len(distance)):
            if all(elem > 30000 for elem in distance[i:i+sequence]):
                if first:
                    start_i = i
                    first=False
                #print(i)
        print(start_i)

        count=0
        #max_sequence = 15
        max_sequence = 30
        for i in range(len(distance)):
            if all(elem > 30000 for elem in distance[i:i+max_sequence]):
                count += 1
                #print(i)
        if count==0:
            b = a[start_i-3:start_i+max_sequence-3]
            c = b.transpose(1,0,2)
            print(c.shape)

            #cut_save_dir = osp.join(
            #    save_dir, "cut_spectrogram", self.target_class)
            #if not os.path.exists(cut_save_dir):
            #    os.makedirs(cut_save_dir)
            file_num=len(os.listdir(self.image_save_dir))+1
            file_name = osp.join(self.image_save_dir, "{0:05d}.png".format(file_num))
            Image_.fromarray(c).save(file_name)
            self.img_msg = self.bridge.cv2_to_imgmsg(c, "rgb8")

        else:
            c = a.transpose(1,0,2)
            print(c.shape)
            self.img_msg = msg
            # file_num=len(os.listdir(self.image_save_dir))+1
            # file_name = osp.join(self.image_save_dir, "{0:05d}.png".format(file_num))
            # Image_.fromarray(c).save(file_name)

        #self.img_msg = self.bridge.cv2_to_imgmsg(c, "rgb8")
        #set stamp
        stamp = msg.header.stamp
        self.img_msg.header.stamp = stamp
        self.cut_spectrogram_pub.publish(self.img_msg)

if __name__ == "__main__":
    cs = CutSpectrogram()
    rospy.spin()


# def hit_spec(target_data, target_class, plot=True):
#     rospack = rospkg.RosPack()
#     save_dir = osp.join(rospack.get_path(
#         "sound_classification"), target_data)
#     image_save_dir = osp.join(
#         save_dir, "original_spectrogram", target_class)
    
#     file_name = osp.join(
#         image_save_dir , "00004.png")
#     #im = Image.open(file_name)
#     #im_list = np.asarray(im)
#     #plt.imshow(im_list)
#     #plt.show()

#     #img = plt.imread(file_name)
#     im = Image.open(file_name)
#     img = np.asarray(im)
#     a = img.transpose(1,0,2)
#     print(a.shape)
#     print(a[40])
#     print(img.shape)

#     distance =[]
#     for i in range(a.shape[0]):
#         dist = 0
#         for j in range(a.shape[1]):
#             dist += (a[i][j][0] - 0)**2 + (a[i][j][1] - 0)**2 + (a[i][j][2] - 127) ** 2
#         distance.append(dist)

#     print(distance)
#     print(len(distance))

#     sequence = 3
#     first = True
#     for i in range(len(distance)):
#         if all(elem > 30000 for elem in distance[i:i+sequence]):
#             if first:
#                 start_i = i
#                 first=False
#             print(i)
#     print(start_i)

#     count=0
#     max_sequence = 15
#     for i in range(len(distance)):
#         if all(elem > 30000 for elem in distance[i:i+max_sequence]):
#             count += 1
#             print(i)
            
#     if count==0:
#         b = a[start_i-max_sequence/2:start_i+max_sequence/2]
#         c = b.transpose(1,0,2)
#         print(c.shape)
#         fig = plt.figure(figsize=(6,6))
#         ax = fig.add_subplot(1,1,1)
#         ax.imshow(c)
#         ax.set_xlabel("time[s]")
#         ax.set_ylabel("frequency[hz]")
#         plt.xticks(np.linspace(0,15,3), np.linspace(0,0.24,3))
#         plt.yticks(np.linspace(0,71,6), np.linspace(4500,0,6))
#         #plt.subplots_adjust(left=0.15, right=0.95, bottom=0, top=1)
#         #if plot:
#         #    plt.show()

#         cut_save_dir = osp.join(
#             save_dir, "cut_spectrogram", target_class)
#         if not os.path.exists(cut_save_dir):
#             os.makedirs(cut_save_dir)
#         file_num=len(os.listdir(cut_save_dir))+1
#         file_name = osp.join(cut_save_dir, "{0:05d}.png".format(file_num))
#         Image.fromarray(c).save(file_name)

#     else:
#         fig = plt.figure(figsize=(6,6))
#         ax = fig.add_subplot(1,1,1)
#         ax.imshow(img)
#         #plt.xticks([0,92],[0, 1.5])
#         #plt.yticks([0,71],[0,4500])
#         plt.xticks(np.linspace(0,93,5), np.linspace(0,1.5,5))
#         plt.yticks(np.linspace(0,71,6), np.linspace(4500,0,6))

#         ax.set_xlabel("time[s]")
#         ax.set_ylabel("frequency[hz]")

#         plt.subplots_adjust(left=0.15, right=0.95, bottom=0, top=1)

#         #if plot:
#         #    plt.show()

#     thesis_save_file_dir = osp.join(
#         save_dir, "thesis", target_class)
#     if not os.path.exists(thesis_save_file_dir):
#         os.makedirs(thesis_save_file_dir)
#     thesis_save_file_name = osp.join(
#         thesis_save_file_dir, "00004.png")
#     plt.savefig(thesis_save_file_name)


#if __name__=="__main__":
#    hit_spec("train_data_grasp", "grasp")
