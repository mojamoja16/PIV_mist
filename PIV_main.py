from openpiv import tools, pyprocess, validation, filters, scaling,piv

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import imageio
import cv2



def PIV_frame(prev_frame,now_frame):
    frame_a  = prev_frame
    frame_b  = now_frame
    
   # fig,ax = plt.subplots(1,2,figsize=(12,10))
    

    winsize = 64 # pixels, interrogation window size in frame A
    searchsize = 81  # pixels, search area size in frame B
    overlap = 8 # pixels, 50% overlap
    dt = 0.01 # sec, time interval between the two frames

    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method='peak2peak',
    )

    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=searchsize,
        overlap=overlap,
    )

    u1, v1, mask = validation.sig2noise_val(
        u0, v0,
        sig2noise,
        threshold = 1.05,
    )
    u2, v2 = filters.replace_outliers(
        u1, v1,
        method='localmean',
        max_iter=3,
        kernel_size=3,
    )

# convert x,y to mm
# convert u,v to mm/sec

    x, y, u3, v3 = scaling.uniform(
        x, y, u2, v2,
        scaling_factor = 96.52,  # 96.52 pixels/millimeter
    )

    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

    tools.save(x, y, u3, v3, mask, 'exp1_001.txt' )

    fig, ax = plt.subplots(figsize=(8,8))
    tools.display_vector_field(
        'exp1.txt',
        ax=ax, scaling_factor=96.52,
        scale=50, # scale defines here the arrow length
        width=0.0035, # width is the thickness of the arrow
        on_img=True, # overlay on the image
        image_name='frame_a',
    );
    

def read_Video():
    video_path = 'test_move5.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    idx = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        cv2.imshow('Video', frame)
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                prev_frame = frame
            elif idx < 10:
                continue
            else:  # 1秒ずつフレームを保存
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                filled_second = str(second).zfill(4)
                now_frame = frame
                #cv2.imshow('Video', frame)
                PIV_frame(prev_frame,now_frame)
                prev_frame = now_frame
                idx = 0
        else:
            break

def Img_proc(frame):
    img_mask = cv2.medianBlur(frame,21)
    
    return img_mask
#read_Video()
video_path = 'mistdetect.avi'
cap = cv2.VideoCapture(video_path)

success,image1 = cap.read()
img1 = Img_proc(image1)
count=0

U = []
V = []
num = 0
while cap.isOpened():
    success,image2 = cap.read()
    img2 = Img_proc(image2)
    if count>=10:
        x,y,u,v = piv.simple_piv(img1.sum(axis = 2),img2.sum(axis = 2),plot = False);
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image1, cmap=plt.get_cmap("gray"), alpha=0.5, origin="upper")
        ax.quiver(x, y, u, -v, scale=70,
                  color='r', width=.005)
        fig.savefig("{}.png".format(num))
        image1 =image2.copy()
        count=0
        U.append(u)
        V.append(v)
    count += 1
    num += 1
        
    
