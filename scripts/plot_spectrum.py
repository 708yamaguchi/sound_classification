#!/usr/bin/env python

from __future__ import division
import rospy
import pylab
import numpy as np
from sound_classification.msg import Spectrum
import matplotlib.pyplot as plt

class SpecPlotNode():
    def __init__(self):
        rospy.Rate(100)
        self.sub_spectrum = rospy.Subscriber(
            "/microphone/sound_spec", Spectrum, self.cb, queue_size=1)

        self.fig = plt.figure(figsize=(15,6))
        self.ax0 = plt.subplot2grid((1,1),(0,0))
        self.ax0.set_title("aaa", fontsize=12)
        self.ax0.set_xlabel("freq", fontsize=12)
        self.ax0.set_ylabel("spec", fontsize=12)
        self.lines0, = self.ax0.plot([-1,-1],[1,1], label="f(x)")

    def cb(self, msg):
        self.frequency = np.array(msg.frequency)
        self.spectrum = np.array(msg.spectrum)

        print("freq=")
        print(len(self.frequency))
        print("cep=")
        print(len(self.spectrum))
        #n=2048
        #plt.plot(self.quefrency*1000, self.cepstrum)
        #plt.xlabel("quefrency")
        #plt.ylabel("log amplitude cepstrum")
        #plt.show()
        #n=2048
        
        self.lines0.set_data(self.frequency[0:256], self.spectrum[0:256])
        self.ax0.set_xlim(0, 5000)
        self.ax0.set_ylim(0,50)
        self.ax0.legend()

if __name__ == "__main__":
    rospy.init_node("specplotnode")
    SpecPlotNode()
    #rospy.spin()
    while not rospy.is_shutdown():
        plt.pause(.0001)
