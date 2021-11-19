import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
from tryyolov4 import Object_detect

# SAVING SYNTAX
# rois : {"cam_num" : [
#                          [bottom_right_x, bottom_right_y, top_left_x, top_left_y]
#                    ]}
rois = {
        '2' : [ [210,255,145,160], [290,255,210,160], [270,170,210,110], [210,180,140,95] ], # done

        '6' : [ [145,250,85,155], [160,300,80,230] ], #done

        '9' : [ [75,240,8,155], [150,280,68,170], [240,270,145,160], [300,230,230,140]],

        '10' : [ [50,300,0,200], [170,255,120,165], [210,225,160,150], [160, 160, 132,110], [140,180,90,120] ],#not done reconsider rois as the table stuff and not the chair? - DONE

        '25' : [ [205,180,160,80], [300,155,265,90], [335,155,295,90] ],    #done
                                                                            # first roi : bench, 
                                                                            # SECOND ROI: OVEN (GET OVEN MAKE IT CHAIR),
                                                                            # also sees a bottle at times have to fix or can ignore - FIXED, 
                                                                            # third roi: tv monitor(good enough) and laptop (yayy)
                }

no_person_rois = { '2' : [ [210,215,150,160], [280,210,215,160], [270,165,220,120], [210,180,140,140] ],
                   '6' : [ [145,250,85,190], [160,265,80,220] ],
                   '9' : [ [77,220,25,160], [100,215,65,170], [240, 210, 200,165] , [265,205,225,155]],
                   '10' : [[40,240,0,190], [165,195,120,150], [190,180,140,140], [170, 170, 132,130], [150,175,95,140]],
                  '25' : [ [195,110,160,80], [300,115,265,90], [335,120,295,90] ]
                  }  # in these rois 'book' is seen as 'bench'
seat_status_indicator = {'empty' : 0, 'occupied' : 1, 'on hold' : 2}

def check_table_roi(cam_num, idx, img):
    table_section = no_person_rois[cam_num][idx]
    df = Object_detect(img[table_section[3]:table_section[1],table_section[2]:table_section[0],:], confThreshold=0.3, nmsThreshold=0.5)
    if df.empty:
        status = 'empty'
    else:
        status = 'on hold'
    return status


# SHOULD BE IN "SEAT STATUS" folder for below function to work properly
def load_images_from_folder(folder):
    images = []
    for sub_folder in os.listdir(folder):
        # if sub_folder == '10_420':
        for filename in os.listdir(folder +'/'+ sub_folder):
            cam_num = filename.split('Camera')[1]
            cam_num = cam_num.split('_')[0]
            if cam_num in ['9']:#['2', '9', '10', '6', '25']: #ground floor cams
                img = cv2.imread(os.path.join(folder, sub_folder, filename))
                if img is not None:
                    # img = cv2.resize(img, (img.shape[1]*2,img.shape[0]*2))
                    roi = rois[cam_num]
                    for idx, chair in enumerate(roi):
                        # Initialize flad and status to default values for each chair
                        flag = 0
                        status = 'empty'
                        print(f"{filename} CAM_NUM: {cam_num} CHAIR : {idx + 1}")
                        print(chair[3],chair[1],chair[2],chair[0])

                        # calling Object_detect on ROI
                        cv2.imshow('img',img[chair[3]:chair[1],chair[2]:chair[0]])
                        cv2.waitKey(0)
                        df = Object_detect(img[chair[3]:chair[1],chair[2]:chair[0],:], confThreshold=0.2, nmsThreshold=0.2)
                        print('in seat status:' ,df)
                        # print(df.empty)
                        # check empty df
                        if df.empty:
                            print("\n\nRECHECKING\n")
                            check_table_roi(cam_num, idx, img)
                        else:
                            # print(df['ClassIds'].values)
                            # print(1 in df['ClassIds'].values)
                            # print('1' in df['ClassIds'].values)
                            if 1 in df['ClassIds'].values:
                                status = 'occupied'
                            else:
                                unique_vals = df['ClassIds'].unique()
                                for item in unique_vals:
                                    if item not in [57,70,71,73,14]:
                                        status = 'on hold'
                                        flag = 1
                                        break
                                
                                if flag == 0:
                                    print("rechecking cuz we got only chair")
                                    check_table_roi(cam_num, idx, img)
                                    
                        print(f"STATUS : {status} {seat_status_indicator[status]}")
                    images.append(img)
    return images
 

folder = 'libimg'
images = load_images_from_folder(folder)

# print(images)
# for image in images: 
#     cv2.imshow('image',image)
#     #resize image with width = 500, and height = 500
#     cv2.imshow('image2',image)
#     cv2.waitKey(0)


'''     DETECTING OBJECTS IN ROIS
with open('COCO_labels.txt','r') as f:
    classIDs = f.read().split('\n')

df = pd.read_csv(f'output_file2.csv')
df = df.drop('Unnamed: 0', axis = 1)

# print(df)

# print(df[df['ClassIds']==56]['TLpoint'])
# print(df[df['ClassIds']==56]['BRpoint'])

tl_points_df = list(df[df['ClassIds']==56]['TLpoint'])
br_points_df = list(df[df['ClassIds']==56]['BRpoint'])

image_name = 'test2_new.jpg'
img = cv2.imread(image_name)


tl_points = []
br_points = []
for tl_point_str, br_point_str in zip(tl_points_df, br_points_df): 
    # getting top left points
    tl_point = tl_point_str.split(', ')
    tl_point[0] = tl_point[0][1:]
    tl_point[1] = tl_point[1][:-1]
    tl_point = list(map(int,tl_point))
    if tl_point[0]<0: tl_point[0] = 0
    if tl_point[1]<0: tl_point[1] = 0
    tl_points.append((tl_point[0],tl_point[1]))

    # getting bottom right points
    br_point = br_point_str.split(', ')
    br_point[0] = br_point[0][1:]
    br_point[1] = br_point[1][:-1]
    br_point = list(map(int,br_point))
    if br_point[0]<0: br_point[0] = 0
    if br_point[1]<0: br_point[1] = 0
    br_points.append((br_point[0],br_point[1]))

# print(tl_points, '\n', br_points, sep = '')

cnt = 0
for tl,br in zip(tl_points, br_points):
    cnt += 1
    print(f'TOP LEFT:\tX: {tl[0]}\t Y: {tl[1]}\tBOTTOM LEFT:\tX: {br[0]}\t Y: {br[1]}')
    
    # plt.imshow(img[tl[0]:br[0], tl[1]:br[1]])
    # x = np.random.randint(200,250)
    # y = np.random.randint(150,200)
    # cv2.rectangle(img, (tl[0],tl[1]),(br[0], br[1]), color = (x - y,255,x), thickness = 2)
    # plt.imshow(img)
    # plt.show()
    
    print(img.shape)
    cv2.imshow('op', img[1:700, 1:100])
    cv2.imshow('output', img[416:563+(563-416),233+(338-233)//2:338+233-(338-233)//2,:])
    # cv2.imshow(f'chair{cnt}',img[tl[0]:br[0], tl[1]:br[1]])
    cv2.waitKey(0)
    cv2.destroyWindow(f'chair{cnt}')

    # cv2.imwrite(f'chair{cnt}.jpg',img[tl[0]:br[0]+1, tl[1]:br[1]+1])
    # if cnt ==5 :
    #     break
'''

'''CODE ENDS'''        

# for i in range(3):
#     print(f' --------------------------------FILE {i}-------------------------------')

#     df = pd.read_csv(f'output_file{i}.csv')
#     df = df.drop('Unnamed: 0', axis = 1)

#     unique_vals = list(df['ClassIds'].unique())

#     for uni in unique_vals:
#         print(f"{classIDs[uni]} : {(df['ClassIds'] == uni).sum()}")
