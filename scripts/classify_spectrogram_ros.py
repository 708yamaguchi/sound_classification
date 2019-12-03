#!/usr/bin/env python
# classify spectrogram using neural networks

import chainer
from chainer import cuda
from chainer_modules import nin

from cv_bridge import CvBridge
import numpy as np
import os.path as osp
from PIL import Image as Image_
import rospy
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import String
#from sound_classification.msg import Probability
from sound_classification.msg import Imagestring
import math


class ClassifySpectrogramROS:
    def __init__(self):
        rospy.init_node('classify_spectrogram_ros')
        archs = {  # only NIN is availale now
            # 'alex': alex.Alex,
            # 'googlenet': googlenet.GoogLeNet,
            # 'googlenetbn': googlenetbn.GoogLeNetBN,
            'nin': nin.NIN,
            # 'resnet50': resnet50.ResNet50,
            # 'resnext50': resnext50.ResNeXt50,
        }

        self.gpu = 0
        #device = chainer.cuda.get_device(self.gpu)  # for python2, gpu number is 0

        #print('Device: {}'.format(device))
        print('Dtype: {}'.format(chainer.config.dtype))
        print('')

        # Initialize the model to train
        rospack = rospkg.RosPack()
        data = rospy.get_param("~train_data", 'train_data')
        n_class_file = osp.join(
            rospack.get_path('sound_classification'),
            data, 'dataset', 'n_class.txt')
        n_class = 0
        self.classes = []
        with open(n_class_file, mode='r') as f:
            for row in f:
                self.classes.append(row.strip())
                n_class += 1
        self.model = archs['nin'](n_class=n_class)
        initmodel = rospy.get_param('~model')
        print('Load model from {}'.format(initmodel))
        chainer.serializers.load_npz(initmodel, self.model)
        if self.gpu >= 0:
            chainer.backends.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        #self.model.to_device(device)
        #device.use()

        # Load the mean file
        mean_file_path = osp.join(rospack.get_path('sound_classification'),
                                  data, 'dataset', 'mean_of_dataset.png')
        self.mean = np.array(Image_.open(mean_file_path), np.float32).transpose(
            (2, 0, 1))  # (256,256,3) -> (3,256,256), rgb

        # Set up an optimizer
        optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
        optimizer.setup(self.model)
        #optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

        # subscriber and publisher
        self.hit_sub = rospy.Subscriber(
            '/microphone/hit_spectrogram', Image, self.hit_cb)
        self.pub = rospy.Publisher(
            '/object_class_by_image', String, queue_size=1)
        self.pub2 = rospy.Publisher(
            "/object", Imagestring, queue_size=1)
        #self.pub2 = rospy.Publisher(
        #    "/probability", Probability, queue_size=1)

        self.bridge = CvBridge()

    def sigmoid(self, x):
        e = math.e
        s = 1 / (1 + e**-x)
        return s

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def hit_cb(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')
        
        with chainer.using_config('train', False), \
             chainer.no_backprop_mode():
            x_data = np.array(Image_.fromarray(cv_image).resize((256, 256))).astype(np.float32)
            x_data = x_data.transpose(
                (2, 0, 1))[[2, 1, 0], :, :]  # (256,256,3) -> (3,256,256), bgr -> rgb
            mean = self.mean.astype(np.float32)
            x_data -= mean
            x_data *= (1.0 / 255.0)  # Scale to [0, 1.0]
            # fowarding once
            x_data = cuda.to_gpu(x_data[None], device=self.gpu)
            #print(np.shape(x_data))
            x_data = chainer.Variable(x_data)
            #print(np.shape(x_data))
            #t_data = np.array([1,0,0,0,0,0,0,0,0,0])
            #t_data = cuda.to_gpu(t_data[None], device=self.gpu)
            #print(t_data.ndim)
            #t_data = chainer.Variable(t_data)
            ret=self.model.forward_for_test(x_data)
            #print(ret.ndim)
            ret = cuda.to_cpu(ret.data)[0]
            #print(ret.ndim)
            #print(type(ret))
            #ret = self.softmax(ret)
            #print(ret)
            #get_msg = String()
            for i in range(len(ret)):
                print(self.classes[i])
                print(ret[i])
                
            #msg2 = Probability()
            #msg2.probability = ret
            #self.pub2.publish(msg2)
        msg3 = Imagestring()
        msg3.header = msg.header
        msg3.height = msg.height
        msg3.width = msg.width
        msg3.encoding = msg.encoding
        msg3.is_bigendian = msg.is_bigendian
        msg3.step = msg.step
        msg3.data = msg.data

        msg2 = String()
        if np.max(ret) > 0.9:
            msg2.data = self.classes[np.argmax(ret)]
            msg3.data2 = self.classes[np.argmax(ret)]
        else:
            msg2.data = "unknown"
            msg3.data2 = "unknown"

        if msg2.data == "robotvoice":
            if np.max(ret) > 0.99:
                msg2.data = "robotvoice"
                msg3.data2 = "robotvoice"
            else:
                msg2.data = "unknown"
                msg3.data2 = "unknown"
        print("-------")
        print(msg2.data)
        print("-------")
        rospy.sleep(0.5)
        #msg.data = 'no object'
        self.pub.publish(msg2)
        self.pub2.publish(msg3)


if __name__ == '__main__':
    csir = ClassifySpectrogramROS()
    rospy.spin()
