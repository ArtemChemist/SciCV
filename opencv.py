# %%
import cv2
from matplotlib import pyplot as plt
import numpy as np

# %%
im = cv2.imread('pos15.jpg', 0)

# %%
plt.imshow(im)

# %%

kernel = np.ones((5,5), dtype = np.float)/25
im_fltd = cv2.medianBlur(im, 3)
edges = cv2.Canny(im, 50,100)
plt.imshow(edges)

# %%
plt.hist(im.flat, 100)
# %%

im_eq = cv2.equalizeHist(im)
plt.imshow(im_eq)
# %%

clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (15,15))
cl_im = clahe.apply(im)
plt.imshow(cl_im)

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
# Otsu thresholding
mask = create_circular_mask(im.shape[0], im.shape[1])
ret, th = cv2.threshold(im[mask], 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
im_th = im.copy()
im_th[im<ret] = 0
im_th[im>ret] = 255
plt.imshow(im_th)

# %%
# Erosion and dilation
kernel = np.ones((5,5), dtype = np.uint8) 
erosion = cv2.erode(im_th, kernel, iterations =1)
dilation = cv2.dilate(erosion, kernel, iterations =1)
plt.imshow(dilation)
# %%
