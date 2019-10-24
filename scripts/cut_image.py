#!/usr/bin/env python

import cv2
import sys
import os

args = sys.argv

if len(args) != 4:
   print("error")
   sys.exit()

class mouseParam:
    def __init__(self, input_file):
        self.mouseEvent = {"x": None, "y": None, "event":None, "flags":None}
        cv2.setMouseCallback(input_file, self._cb, None)

    def _cb(self, eventType, x, y, flags, userdata):
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    def getEvent(self):
        return self.mouseEvent["event"]

    def getData(self):
        return self.mouseEvent

    def getX(self):
        return self.mouseEvent["x"]

    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])

# def cut_img(label, input_file, output_file):
#     #print(os.path.join("../train_data/original_spectrogram", label))
#     path = os.path.join("../train_data/original_spectrogram", label)
#     #img = cv2.imread("../train_data/original_spectrogram/table/00001.png")
#     #print(img.shape)
#     img = cv2.imread(os.path.join(path, input_file))
#     print(img.shape)

#     cut_start = int(args[1])
#     cut_end = int(args[2])
#     clipped_img = img[:, cut_start:cut_end, :]
#     cv2.imwrite(os.path.join(path, output_file), clipped_img)
#     print(clipped_img.shape)

if __name__=="__main__":
    #cut_img(label=args[3], input_file=args[4], output_file=args[5])
    label = args[1]
    input_file = args[2]
    output_file = args[3]

    path = os.path.join("/home/nakaotatsuya/audio_ws/src/sound_classification/train_data/original_spectrogram", label)
    read = cv2.imread(os.path.join(path, input_file))
    window_name="input window"
    cv2.imshow(window_name, read)
    mouseData = mouseParam(window_name)

    cut_start = 0
    cut_end = 0
    while 1:
        cv2.waitKey(20)
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            print(mouseData.getPos())
            cut_start = int(mouseData.getX())
            #print(cut_start)

        if mouseData.getEvent() == cv2.EVENT_LBUTTONUP:
            cut_end = int(mouseData.getX())
            #print(cut_end)

        elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            break
    cv2.destroyAllWindows()
    print("Fininsh")

    clipped_img = read[:, cut_start:cut_end, :]
    cv2.imwrite(os.path.join(path, output_file), clipped_img)
    print(clipped_img.shape)

