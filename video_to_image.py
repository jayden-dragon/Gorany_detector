import cv2
import os
from matplotlib import pyplot as plt

vidcap = cv2.VideoCapture('/Users/msi/Desktop/yolo3/gorany.mp4')

count = 0

while(vidcap.isOpened()):

    ret, img = vidcap.read()
 
    cv2.imwrite("/Users/msi/Desktop/yolo3/input/%d.jpg" % count, img)
 
    print('Saved %d.jpg' % count)
    count += 1
 
    vidcap.release()