from IPython.display import display
from PIL import Image
from yolo import YOLO
import cv2
import os
from matplotlib import pyplot as plt

vidcap = cv2.VideoCapture('/Users/msi/Desktop/yolo3/.mp4')

count = 0

while(vidcap.isOpened()):

    ret, img = vidcap.read()
 
    cv2.imwrite("/Users/msi/Desktop/yolo3/input/%d.jpg" % count, img)
 
    print('Saved frame%d.jpg' % count)
    count += 1
 
    vidcap.release()

def objectDetection(file, model_path, class_path):
    yolo = YOLO(model_path=model_path, classes_path=class_path, anchors_path='model_data/tiny_yolo_anchors.txt')
    image = Image.open(file)
    result_image = yolo.detect_image(image)
    display(result_image)

input = os.listdir('/Users/msi/Desktop/yolo3/input/')[:]
input.sort()

for i in range(len(input)):
    objectDetection('', 'model_data/yolo_tiny.h5', './model_data/coco_classes.txt')
    result_image.save('/Users/msi/Desktop/yolo3/output/result%d.jpg' %i)