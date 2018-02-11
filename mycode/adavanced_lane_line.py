import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

cam_param = pickle.load(open("cam_param.p", "rb"))
print(cam_param)

mtx = cam_param["mtx"]
dist = cam_param["dist"]


# Read in an image
img = cv2.imread('../camera_cal/calibration1.jpg')
cv2.imshow("img",img)
cv2.waitKey()

undistorted = cv2.undistort(img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()