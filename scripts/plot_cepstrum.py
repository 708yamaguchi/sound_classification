#!/usr/bin/env python

from __future__ import division
import rospy
import pylab
import numpy as np
from sound_classification.msg import Cepstrum
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

class CepsPlotNode():
    def __init__(self):
        self.UPDATE_SECOND=10
        self.quefrency = []
        self.cepstrum = []
        #qt graph
        self.app = QtGui.QApplication([])
        self.app.quitOnLastWindowClosed()
        #window
        self.win = QtGui.QMainWindow()
        self.win.setWindowTitle("SpectrumVisualizer")
        self.win.resize(750,400)
        self.centralwid = QtGui.QWidget()
        self.win.setCentralWidget(self.centralwid)
        #Layout
        self.lay = QtGui.QVBoxLayout()
        self.centralwid.setLayout(self.lay)

        self.plotwid1 = pg.PlotWidget(name="spectrum")
        self.plotitem1 = self.plotwid1.getPlotItem()
        self.plotitem1.setMouseEnabled(x=False, y=False)
        self.plotitem1.setXRange(0, 0.0025, padding=0)
        self.plotitem1.setYRange(-2, 2)
        self.specAxis1 = self.plotitem1.getAxis("bottom")
        self.specAxis1.setLabel("Frequency")
        self.curve1 = self.plotitem1.plot()
        self.lay.addWidget(self.plotwid1)

        #show plot
        self.win.show()
        #update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.UPDATE_SECOND)
        
        rospy.Rate(100)
        self.sub_cepstrum = rospy.Subscriber(
            "/microphone/cepstrum", Cepstrum, self.cb, queue_size=1)

        #self.fig = plt.figure(figsize=(15,6))
        #self.ax0 = plt.subplot2grid((1,1),(0,0))
        #self.ax0.set_title("aaa", fontsize=12)
        #self.ax0.set_xlabel("que", fontsize=12)
        #self.ax0.set_ylabel("ceps", fontsize=12)
        #self.lines0, = self.ax0.plot([-1,-1],[1,1], label="f(x)")

    def cb(self, msg):
        self.quefrency = np.array(msg.quefrency)
        self.cepstrum = np.array(msg.cepstrum)

        print("que=")
        print(len(self.quefrency))
        print("cep=")
        print(len(self.cepstrum))
        #n=2048
        #plt.plot(self.quefrency*1000, self.cepstrum)
        #plt.xlabel("quefrency")
        #plt.ylabel("log amplitude cepstrum")
        #plt.show()
        #n=2048
        #self.lines0.set_data(self.quefrency, self.cepstrum)
        #self.ax0.set_xlim((self.quefrency.min(), self.quefrency.max()+0.01))
        #self.ax0.set_ylim((-30,30))
        #self.ax0.legend()

    def update(self):
        self.curve1.setData(self.quefrency, self.cepstrum)

if __name__ == "__main__":
    rospy.init_node("cepsplotnode")
    CepsPlotNode()
    try:
        QtGui.QApplication.instance().exec_()
    except:
        print("terminate")
    #rospy.spin()
    #while not rospy.is_shutdown():
    #    plt.pause(.0001)
