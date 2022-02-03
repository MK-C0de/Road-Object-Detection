import torch
model0 = torch.hub.load('ultralytics/yolov5', 'yolov5s') # or yolov5m, yolov5l, yolov5x, custom
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
model2 = torch.hub.load('./yolov5/', 'custom', path='./yolov5x_street_1/weights/best.pt', source='local')
model3 = torch.hub.load('./yolov5/', 'custom', path='./yolov5x_street/weights/best.pt', source='local')

import sys
import cv2
import numpy as np

import skimage.exposure as exposure
import matplotlib.pyplot as plt

from importlib import reload
import utils; reload(utils)
from utils import *

import warnings
warnings.filterwarnings('ignore')

def get_stoplight_color(image, xy0, xy1):
    x0,y0 = xy0
    x1,y1 = xy1
    
    roi = image[y0:y1, x0:x1]
    histGR = cv2.calcHist([roi], [1, 2], None, [256, 256], [0, 256, 0, 256])

    color = ('b','g','r')
    for k,color in enumerate(color):
        histogram = cv2.calcHist([roi],[k],None,[256],[0,256])
        plt.plot(histogram,color = color)
        plt.xlim([0,256])
        plt.show()

    histScaled = exposure.rescale_intensity(histGR, in_range=(0,1), out_range=(0,255)).clip(0,255).astype(np.uint8)

    # make masks
    ww, hh = 256, 256
    ww13, hh13 = ww // 3, hh // 3
    ww23, hh23 = 2 * ww13, 2 * hh13
    black = np.zeros_like(histScaled, dtype=np.uint8)

    # specify points in OpenCV x,y format
    ptsUR = np.array( [[[ww13,0],[ww-1,hh23],[ww-1,0]]], dtype=np.int32 )
    redMask = black.copy()
    cv2.fillPoly(redMask, ptsUR, (255,255,255))
    ptsBL = np.array( [[[0,hh13],[ww23,hh-1],[0,hh-1]]], dtype=np.int32 )
    greenMask = black.copy()
    cv2.fillPoly(greenMask, ptsBL, (255,255,255))

    region = cv2.bitwise_and(histScaled,histScaled,mask=redMask)
    redCount = np.count_nonzero(region)
    region = cv2.bitwise_and(histScaled,histScaled,mask=greenMask)
    greenCount = np.count_nonzero(region)

    threshCount = 45
    if redCount > greenCount and redCount > threshCount: return 0
    elif greenCount > redCount and greenCount > threshCount: return 1
    elif redCount < threshCount and greenCount < threshCount: return None
    else: return None

def detect_car (frame, isBest):
    result_img = np.copy(frame)

    if isBest: results = model(result_img)
    else: results = model0(result_img)

    for index, row in enumerate(results.xyxy[0]):
        row = row.detach().numpy()

        x0,y0,x1,y1,conf,c = row
        xy0 = np.array([x0,y0], dtype=int)
        xy1 = np.array([x1,y1], dtype=int)

        if conf < .60: continue

        if c == 9:
            # Stoplight
            guess_color = get_stoplight_color(frame, xy0, xy1)
            if guess_color is None:
                cv2.rectangle(np.copy(frame), xy0, xy1, (0,0,0), -1)
            else:
                colors = [(0,0,255), (0,255,0), (0,255,255)]
                color = colors[guess_color]
                cv2.rectangle(result_img, xy0, xy1, color, -1)
            cv2.rectangle(result_img, xy0, xy1, (0,0,255), 2)
            cv2.putText(result_img, "TRAFFIC LIGHTS" ,xy0,cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1, cv2.LINE_AA)
        elif c == 2 or c == 7:
            cv2.rectangle(result_img, xy0, xy1, (255,255,255), 2)
            cv2.putText(result_img, "CAR" ,xy0,cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(result_img, xy0, xy1, (0,255,0), 2)
    return result_img

def detect_signs (frame):
    result_img = np.copy(frame)
    results = model3(result_img)

    for index, row in enumerate(results.xyxy[0]):
        row = row.detach().numpy()

        x0,y0,x1,y1,conf,c = row
        xy0 = np.array([x0,y0], dtype=int)
        xy1 = np.array([x1,y1], dtype=int)

        if conf < .45: continue

        if c == 0:
            cv2.rectangle(result_img, xy0, xy1, (0,0,255), 2)
            cv2.putText(result_img, "TRAFFIC LIGHTS" ,xy0,cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1, cv2.LINE_AA)
        elif c == 1:
            cv2.rectangle(result_img, xy0, xy1, (255,0,0), 2)
            cv2.putText(result_img, "STOP" ,xy0,cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        elif c == 2:
            cv2.rectangle(result_img, xy0, xy1, (255,0,0), 2)
            cv2.putText(result_img, "SPEED" ,xy0,cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(result_img, xy0, xy1, (0,255,0), 2)
    return result_img

def main(argv):
    print(cv2.__version__)

    # import video file and video capture it
    video = cv2.VideoCapture('./Driving Footage/night_stop.mp4')
    if not video.isOpened(): print('Error Opening Video')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output1 = cv2.VideoWriter('yolov5x1-night-stop.mp4', fourcc, 15.0, (1280, 720))
    output2 = cv2.VideoWriter('yolov5x20-night-stop.mp4', fourcc, 15.0, (1280, 720))

    while True:
        success, frame = video.read()
        if success: 
            # standardize, recolor and crop the frame
            height, width = frame.shape[0], frame.shape[1]
            copy_frame = np.copy(frame)
            
            car_base = detect_car(copy_frame, False)
            car_best = detect_car(copy_frame, True)

            best = detect_signs(car_best)

            cv2.imshow('', best)
            if cv2.waitKey(1) == ord('q'): break
            output1.write(car_base)
            output2.write(best)
        else: break
    video.release()
    output1.release()
    output2.release()
    cv2.destroyAllWindows()
    print('video processed')

if __name__ == '__main__':
    main(sys.argv[1:])