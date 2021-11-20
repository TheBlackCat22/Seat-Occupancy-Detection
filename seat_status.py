import cv2
import pandas as pd
# import matplotlib.pyplot as plt - used to make rois

import os
from object_detection import Object_detect

# CONVENTION
# rois : {"cam_num" : [
#                          [bottom_right_x, bottom_right_y, top_left_x, top_left_y]
#                    ]}
rois = {
        '2' : [ [210,255,145,160], [290,255,210,160], [270,170,210,110], [210,180,140, 85] ],

        '6' : [ [145,250,85,155], [160,300,80,230] ],

        '9' : [ [75,240,8,155], [150,280,68,170], [130,175,80,120], [70,168,30,127], [240,270,145,160], [300,230,230,140], [290,165,215,115]], # table-1: 1,2,3,4 ; table 2: 5,6,8

        '10' : [[170,255,120,165], [210,225,160,150], [160, 160, 132,110], [140,180,90,120] ],

        '25' : [ [205,180,160,80], [300,155,265,90], [335,155,295,90] ],    
                }

# if the model only detects chairs or detects nothing at all then zoom in (search in these rois)
no_person_rois = { '2' : [ [210,215,150,160], [280,210,215,160], [270,165,220,120], [210,180,140,140] ],
                   '6' : [ [145,250,85,190], [160,265,80,220] ],
                   '9' : [ [77,220,25,160], [100,215,65,170], [105,170,75,140], [80,165,51,142], [240, 210, 200,165] , [265,205,225,155], [248,165,214,130], ], # table-2: 7 was unobservable.
                   '10' : [ [165,195,120,150], [190,180,140,140], [170, 170, 132,130], [150,175,95,140]], 
                  '25' : [ [195,110,160,80], [300,115,265,90], [335,120,295,90] ]
                  }  

# mapping seat status to a value
seat_status_indicator = {'empty' : 0, 'occupied' : 1, 'on hold' : 2}

# check in the smaller rois (only called when no object is detected or only chair is detected in the big ROI)
def check_table_roi(cam_num, idx, img):
    table_section = no_person_rois[cam_num][idx]
    df = Object_detect(img[table_section[3]:table_section[1],table_section[2]:table_section[0],:], confThreshold=0.3, nmsThreshold=0.5)
    if df.empty:
        status = 'empty'
    else:
        status = 'on hold'
    return status

# detect images in folder 
def load_images_from_folder(folder):
    final_df = pd.DataFrame(columns = ['Camera Number', 'Chair Number', 'Status' ])
    for filename in os.listdir(folder):
        cam_num = filename.split('Camera')[1]
        cam_num = cam_num.split('_')[0]
        if cam_num in ['2','6','9','10','25']: #ground floor cams
            img = cv2.imread(os.path.join(folder, filename))
            print(f"Camera Number: {cam_num}\t ROI number : {idx + 1} is being processed.")

            # resize image - rois are defined on this size
            img = cv2.resize(img , (352, 288)) 

            # check if image is empty
            if img is not None:
                roi = rois[cam_num]

                for idx, chair in enumerate(roi):

                    # Initialize flag and status to default values for each chair
                    flag = 0
                    status = 'empty'
                    
                    # calling Object_detect on ROI
                    df = Object_detect(img[chair[3]:chair[1],chair[2]:chair[0],:], confThreshold=0.3, nmsThreshold=0.5)

                    # check if df is empty
                    if df.empty:
                        # print("\n\nRECHECKING\n")
                        status = check_table_roi(cam_num, idx, img)
                    else:
                        if 1 in df['ClassIds'].values:
                            status = 'occupied'
                        else:
                            unique_vals = df['ClassIds'].unique()
                            for item in unique_vals:
                                if item not in [57,69,70,71,73,14]: # ignore chairs ( all these items have been mapped to chairs, manually by us )
                                    status = 'on hold'
                                    flag = 1
                                    break
                            
                            if flag == 0:
                                # print("rechecking because we got only chair")
                                status = check_table_roi(cam_num, idx, img)
                    print(f"Status : {status}\n")
                    final_df = final_df.append({'Camera Number' : cam_num, 'Chair Number' : idx + 1, 'Status' : seat_status_indicator[status]}, ignore_index=True)
    print("Instance complete")
    return final_df

# change folder name to: 'f1', 'f2', 'f3' to test other images
folder_name = 'f3'
final_df = load_images_from_folder(folder_name)

# changing type to 'int' so as to sort by camera number
final_df = final_df.astype(int)
final_df = final_df.sort_values(by=['Camera Number', 'Chair Number'], ascending=True)

# save final_df in a csv file
final_df.to_csv('seat_status.csv', index = False)

'''
                        RESULTS:
Folder f1: 15/20
Folder f2: 15/20
Folder f3: 16/20
Folder f4: 15/20
'''