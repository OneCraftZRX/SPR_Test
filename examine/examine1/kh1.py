import ncsyproclib as mylib
import numpy
import cv2
import os
print(os.getcwd())
img=cv2.imread(r"C:\Users\25176\OneDrive\Codes\SPR2024\TestGit\examine\examine1\1.bmp")
binary=mylib.ncsyprocs.open_with_threshold(img,1,175,8)
# binary=mylib.ncsyprocs.close_with_threshold(img,1,185,6)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    cv2.drawContours(img, contours[i], -1, (0, 255, 0), 1)  # 改变的是img这张图
    cv2.imshow('findedge', img)
    cv2.waitKey(0)
cv2.drawContours(img, contours, -1, (0, 255, 0), 1)  # 改变的是img这张图
cv2.imshow('findedge', img)
cv2.waitKey(0)