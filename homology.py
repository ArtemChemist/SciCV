# %%
import cv2
from matplotlib import pyplot as plt
import numpy as np

# %%
im1 = cv2.imread('pos15.jpg')
gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

im2 = cv2.imread('pos16.jpg')
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

gray1 = np.float32(gray1)
gray2 = np.float32(gray2)
# %%
def create_circular_mask(h, w, center=None, radius=None):
    '''
    Creates a mask of dimentions, height = h, width = x, with a circle marked true, 
    located at the center and having radius  = radius
    I modified it from here:
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    '''
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    #Compute square of the radius to avoid computing sqrt on every step
    radius_sq = radius**2

    Y, X = np.ogrid[:h, :w]
    dist_from_center_sq = (X - center[0])**2 + (Y-center[1])**2

    mask = dist_from_center_sq <= radius_sq

    return mask

# %%
orb = cv2.ORB_create(100)

msk_out = create_circular_mask(gray1.shape[0], gray1.shape[1], radius=0.9*gray1.shape[1]/2)
msk_in = create_circular_mask(gray1.shape[0], gray1.shape[1], radius=0.8*gray1.shape[1]/2)
msk = msk_out & ~msk_in
im_work = im1.copy()
im_work[~msk] = [0,0,0]
kp1, des1 = orb.detectAndCompute(im_work, None)


msk_out2 = create_circular_mask(gray2.shape[0], gray2.shape[1], radius=0.9*gray2.shape[1]/2)
msk_in2 = create_circular_mask(gray2.shape[0], gray2.shape[1], radius=0.8*gray2.shape[1]/2)
msk2 = msk_out2 & ~msk_in2
im_work = im2.copy()
im_work[~msk2] = [0,0,0]
kp2, des2 = orb.detectAndCompute(im_work, None)
# %%
img1 = cv2.drawKeypoints(im1, kp1, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img1)


# %%
img2 = cv2.drawKeypoints(im2, kp2, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img2)
# %%
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(des1, des2, None)
# %%
matches = sorted(matches, key = lambda x: x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None)
plt.imshow(img3)
# %%
points1 = np.zeros((len(matches), 2), dtype = np.float32)
points2 = np.zeros((len(matches), 2), dtype = np.float32)

for i, match in enumerate(matches):
    points1[i,:] = kp1[match.queryIdx].pt
    points2[i,:] = kp1[match.trainIdx].pt

h, msk = cv2.findHomography(points1, points2, cv2.RANSAC)

hight, width, chan = im2.shape

im1Reg = cv2.warpPerspective(im1, h, (width, hight), cv2.WARP_INVERSE_MAP)

# %%
plt.imshow(im1Reg)
# %%
