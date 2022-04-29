# %%
import cv2
from matplotlib import pyplot as plt
import numpy as np

# %%
im = cv2.imread('pos15.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# %%
plt.imshow(im)
# %%
gray = np.float32(gray)
# %%
harris = cv2.cornerHarris(gray, 15, 3, 0.008)
harris = 255*harris/harris.max()
plt.imshow(harris)
# %%
#Another algorythm for corner detection
plt.imshow(im)
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
for i in corners:
    x,y = i.ravel()
    cv2.circle(im, (x,y), 3,255,-1)
plt.imshow(im)


# %%
orb = cv2.ORB_create(20)
kp, des = orb.detectAndCompute(gray, None)
img2 = cv2.drawKeypoints(im, kp, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img2)
# %%
