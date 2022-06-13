import cv2
import numpy as np




def drawOpticalFlow(frame2, flow, step):
    h, w = frame2.shape[:2]
    x, y = np.mgrid[0:w:step, 0:h:step].reshape(2, -1).astype(np.int32)
    dx, dy = flow[y, x].T
    dist = np.sqrt(dx**2 + dy**2)

    # whereを用いて表示する移動量の閾値を決定 -> threshold
    threshold = 1
    index = np.where(threshold < dist)
    x, y = x[index], y[index]
    dx, dy = dx[index], dy[index]
    line = np.vstack([x, y, x + dx, y + dy]).T.reshape(-1, 2, 2).astype(np.int32)

    # 結果描写
    result = cv2.polylines(frame2, line, False, (0, 0, 255))
    return result
print(type(cv2.getBuildInformation()))
cap = cv2.VideoCapture("testVideo.MTS")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#フレームレート取得
fps = cap.get(cv2.CAP_PROP_FPS)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('output22.mp4', fmt, fps, (width, height))

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    #hsv[...,0] = ang*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    opt_frame = drawOpticalFlow(frame2, flow, 16)
    
    #cv2.imshow('frame2',opt_frame)
    writer.write(opt_frame)
    k = cv2.waitKey(30) & 0xff
    
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',opt_frames)
        
    prvs = next

cap.release()
writer.release()
cv2.destroyAllWindows()