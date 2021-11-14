import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
from tryyolov4 import Object_detect


#                                                  TESTING
# img = cv2.imread('library nvr_IP Camera25_library nvr_20211110173721_7770227.bmp')
# cv2.rectangle(img, (205,180), (165,90), (255,255,255), 1)
# plt.imshow(img)
# plt.show()


# SAVING SYNTAX
# rois : {"cam_num" : [
#                          [bottom_right_x, bottom_right_y, top_left_x, top_left_y]
#                    ]}
rois = {
        '2' : [ [210,255,145,180], [290,255,215,180], [270,175,210,105], [200,175,150,105] ],
        '6' : [ [150,220,85,155], [150,275,85,240] ],
        # '9' : [ [500,500],[600,600]],
        '10' : [ [50,300,0,200], [170,255,120,165], [210,225,160,150], [160, 160, 132,110], [130,180,85,120] ],
        '25' : [ [205,180,165,90], [295,180,260,98], [335,175,300,95] ],
                }

#                                           TESTING
# image = cv2.imread('library nvr_IP Camera25_library nvr_20211110173721_7770227.bmp')
# print(image.shape)
# cv2.imshow('img',image)
# cv2.waitKey(0)

# #1
# cv2.imshow('image_cropped_chair_1', image[90:180,166:204,:])
# cv2.waitKey(0)
# df = Object_detect(image[90:180,166:204,:])
# print(df)

# #2
# cv2.imshow('image_cropped_chair_2', image[98:180,261:294,:])
# cv2.waitKey(0)
# df = Object_detect(image[98:180,261:294,:])
# print(df)

# #3
# cv2.imshow('image_cropped_chair_3', image[95:175,301:334,:])
# cv2.waitKey(0)
# df = Object_detect(image[95:175,301:334,:])
# print(df)



# SHOULD BE IN "SEAT STATUS" folder for below function to work properly
def load_images_from_folder(folder):
    images = []
    for sub_folder in os.listdir(folder):
        # if sub_folder == '10_420':
        for filename in os.listdir(folder +'/'+ sub_folder):
            cam_num = filename.split('Camera')[1]
            cam_num = cam_num.split('_')[0]
            if cam_num in ['2','6','10','25']: #ground floor cams
                img = cv2.imread(os.path.join(folder, sub_folder, filename))
                if img is not None:
                    # img = cv2.resize(img, (img.shape[1]*2,img.shape[0]*2))
                    roi = rois[cam_num]
                    for chair in roi:
                        print(f"CHAIR : {roi.index(chair)}")
                        print(chair[3],chair[1],chair[2],chair[0])

                        # # whole image
                        # df = Object_detect(img)
                        # print(df)
                        
                        # cropped image
                        df = Object_detect(img[chair[3]:chair[1],chair[2]:chair[0],:], confThreshold=0.2, nmsThreshold=0.2)
                        print(df)
                        
                        # cv2.rectangle(img, (chair[2],chair[3]), (chair[0],chair[1]), (255,255,255), 1)
                        # cv2.imshow('img', img)
                        # cv2.waitKey(0)

                        # plt.imshow(img)
                        # plt.show()
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


'''             DETECTING OBJECTS IN ROIS
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
