import numpy as np
from cv2 import cv2
import pandas as pd

'''
classes=[]
with open("Object Detection\COCO_labels.txt","r") as f:
    classes= f.read().split("\n")
'''

def Object_detect(frame,confThreshold=0.4,nmsThreshold=0.3):
    #Download weights yolov4-pg.weights from https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p6.weights
    net=cv2.dnn.readNet("yolov4-p6.cfg","yolov4-p6.weights")
    outNames = net.getUnconnectedOutLayersNames()
    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        '''
        def drawPred(classId, conf, left, top, right, bottom):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

            label = '%.2f' % conf

            if classes:
                assert(classId < len(classes))
                label = '%s: %s' % (classes[classId], label)

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        '''

        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId+1) #correct class ID
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        global df
        df=pd.DataFrame(columns=["ClassIds","Confidences","TLpoint","BRpoint"]) 
        
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            df=df.append({"ClassIds":classIds[i],"Confidences":confidences[i],"TLpoint":[left, top],"BRpoint":[left + width, top + height]},ignore_index=True)
            #drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    blob=cv2.dnn.blobFromImage(frame,1/255,(1280,1280),(0,0,0),swapRB=True,crop=False)

    net.setInput(blob)
    outs=net.forward(outNames)
    postprocess(frame,outs)

    return(df)

'''
img = cv2.imread('IP Camera9_library nvr_library nvr_20211106131741_20211106135959_1175882.jpg')
print(img.shape)
img = cv2.resize(img , (352, 288))
# cv2.imshow('image',img)
# cv2.waitKey(0)

import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()

chair = [300,230,230,140] #[RIGHT, BOTTOM, LEFT, TOP]
cv2.imshow('img',img)
cv2.waitKey(0)
# 190, 165, 200, 210

img = img[chair[3]:chair[1],chair[2]:chair[0]]

cv2.imshow('img2',img)
cv2.waitKey(0)

df = Object_detect(img, confThreshold=0.3, nmsThreshold=0.5)
print(df)
'''


'''
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()

chair =[265,205,225,155] #[RIGHT, BOTTOM, LEFT, TOP]
cv2.imshow('img',img)
cv2.waitKey(0)

img = img[chair[3]:chair[1],chair[2]:chair[0]]

cv2.imshow('img2',img)
cv2.waitKey(0)

df = Object_detect(img, confThreshold=0.3, nmsThreshold=0.5)
print(df)
'''


# 160:240, 142:205
# img2  = cv2.imread('SodaPDF-converted-library nvr_IP Camera2_library nvr_20211110161949_3117577.jpg')
# cv2.imshow('img2',img2)
# cv2.waitKey(0)
# df2 = Object_detect(img2[85:175, 130:210], confThreshold=0.1, nmsThreshold=0.5)
# print(df)