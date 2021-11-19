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
        '2' : [ [210,255,145,160], [290,255,210,160], [270,170,210,110], [210,180,145,90] ],

        '6' : [ [145,250,85,155], [160,300,80,230] ],

        '9' : [ [75,240,8,155], [150,280,68,170], [240,270,145,160], [300,230,230,140]], # make 4 more chairs

        '10' : [[170,255,120,165], [210,225,160,150], [160, 160, 132,110], [140,180,90,120] ],

        '25' : [ [205,180,160,80], [300,155,265,90], [335,155,295,90] ],    
                }

no_person_rois = { '2' : [ [210,215,150,160], [280,210,215,160], [270,165,220,120], [210,180,140,140] ],
                   '6' : [ [145,250,85,190], [160,265,80,220] ],
                   '9' : [ [77,220,25,160], [100,215,65,170], [240, 210, 200,165] , [265,205,225,155]], # make 4 more table sections
                   '10' : [ [165,195,120,150], [190,180,140,140], [170, 170, 132,130], [150,175,95,140]], 
                  '25' : [ [195,110,160,80], [300,115,265,90], [335,120,295,90] ]
                  }  
seat_status_indicator = {'empty' : 0, 'occupied' : 1, 'on hold' : 2}

def check_table_roi(cam_num, idx, img):
    table_section = no_person_rois[cam_num][idx]
    df = Object_detect(img[table_section[3]:table_section[1],table_section[2]:table_section[0],:], confThreshold=0.3, nmsThreshold=0.5)
    if df.empty:
        status = 'empty'
    else:
        status = 'on hold'
    return status

final_df = pd.DataFrame(columns = ['Camera Number', 'Chair Number', 'Status' ])

# SHOULD BE IN "SEAT STATUS" folder for below function to work properly
def load_images_from_folder(folder):
    global final_df
    for sub_folder in os.listdir(folder):
        for filename in os.listdir(folder +'/'+ sub_folder):
            cam_num = filename.split('Camera')[1]
            cam_num = cam_num.split('_')[0]
            if cam_num in ['2', '9', '10', '6', '25']: #ground floor cams
                img = cv2.imread(os.path.join(folder, sub_folder, filename))
                
                # resize image - rois are defined on this size
                img = cv2.resize(img , (352, 288)) 
                if img is not None:
                    roi = rois[cam_num]

                    for idx, chair in enumerate(roi):

                        # Initialize flag and status to default values for each chair
                        flag = 0
                        status = 'empty'
                        print(f"{filename} CAM_NUM: {cam_num} CHAIR : {idx + 1}")
                        print(chair[3],chair[1],chair[2],chair[0])

                        # calling Object_detect on ROI
                        cv2.imshow('img',img[chair[3]:chair[1],chair[2]:chair[0]])
                        cv2.waitKey(2)
                        df = Object_detect(img[chair[3]:chair[1],chair[2]:chair[0],:], confThreshold=0.2, nmsThreshold=0.2)
                        print('in seat status:' ,df)

                        # check if df is empty
                        if df.empty:
                            print("\n\nRECHECKING\n")
                            status = check_table_roi(cam_num, idx, img)
                        else:
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
                                    status = check_table_roi(cam_num, idx, img)
                                    
                        print(f"STATUS : {status} {seat_status_indicator[status]}")
                        final_df = final_df.append({'Camera Number' : cam_num, 'Chair Number' : idx + 1, 'Status' : seat_status_indicator[status]}, ignore_index=True)
                    

 

folder = 'libimg'
load_images_from_folder(folder)

# save final_df in a csv file
final_df.to_csv('../web integration/seat_status.csv', index = False)

