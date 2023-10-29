import easyocr
import cv2
import numpy as np
reader = easyocr.Reader(['ch_sim','en'], gpu = True) # need to run only once to load model into memory

def makevideo_ocr(videoinpath,videooutpath):
        # videoinpath = 'video.mp4'
        # videooutpath = 'video_out.mp4'
        capture = cv2.VideoCapture(videoinpath)
        fourcc = cv2.VideoWriter_fourcc(*'X265')
        writer = cv2.VideoWriter(videooutpath ,fourcc, 24.0, (640,512), True)
        if capture.isOpened():
            while True:
                ret,img_src=capture.read()
                if not ret:break
                # img_out = 
                
                # writer.write(img_out)
                cv2.imshow('Processing',img_src)
                cv2.waitKey(1)
                print(reader.readtext(img_src, allowlist ='0123456789',detail=0,min_size=250))
        else:
            print('视频打开失败！')
        writer.release()

makevideo_ocr("./testV.mp4","./testV_out.mp4")
print("over")