# %%
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import color, io, measure
from scipy import ndimage
import pandas as pd

#%% Read data
img = cv2.imread('pos15.jpg') 
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

# %% Otsu thresholding, followed by openeing
msk_circ = create_circular_mask(im.shape[0], im.shape[1])
ret, th = cv2.threshold(im[msk_circ], 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
im_th = im.copy()
im_th[im<ret] = 0
im_th[im>ret] = 255
eroded = cv2.erode(im_th, np.ones((5,5), np.uint8), iterations = 1)
dilated = cv2.dilate(eroded, np.ones((5,5), np.uint8), iterations = 1)
plt.imshow(dilated )

# %% FIND DEFINITE NOT BACKGROUND: >Threshold + margin
sure_not_bg = cv2.dilate(dilated, np.ones((5,5), np.uint8), iterations = 2)
plt.imshow(sure_not_bg, cmap="gray")

# %% FIND DEFINITE FOREGROUND: far from backgr pixels
# Distance transfrom: how far each pixel to the nearest 0 value pixel
dist_transform = cv2.distanceTransform(dilated, cv2.DIST_L2, 3)
#Threshold it, i.e. only take pixels that are far away from nearest 0
#sure_fg = sure that it is foreground
ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255,0)
sure_fg = np.uint8(sure_fg)
plt.imshow(sure_fg, cmap="gray")

#%% FIND REGION THAT NEEDS SEGMENTING
unknown = cv2.subtract(sure_not_bg, sure_fg)
plt.imshow(unknown, cmap='gray')

#%% Break up regions that might be only loosely connected
ret3, markers = cv2.connectedComponents( sure_fg  )
# Markers - an image, where each pixel has a value equal to the id of the region it belongs to
plt.imshow(markers, cmap="gray")

#%% WTERSHED: but first , do some prep
markers = markers + 10 #raise all markers by random value
# set all regions that need segmenting to background
markers [unknown ==255] = 0
# Now we have an image, where all regios that we are sure about are already marked
# and everything else is 0
# Now do the watershed
markers = cv2.watershed(img, markers) #Does watershed, sets boundaries to -1
img[markers == -1] = [0,0,255] # Make boundaries visible
img2 = color.label2rgb(markers, bg_label=0) # Make a color-labeld image
plt.imshow(img2)

# %% Genrate clusters
clusters = measure.regionprops(label_image = markers, intensity_image= im)

# %% Prep conversion to Pandas
def scalar_attributes_list(im_props):
    """
    Makes list of all scalar, non-dunder, non-hidden
    attributes of skimage.measure.regionprops object
    """
    
    attributes_list = []
    
    for i, test_attribute in enumerate(dir(im_props[0])):
        
        #Attribute should not start with _ and cannot return an array
        #does not yet return tuples
        if test_attribute[:1] != '_' and not\
                isinstance(getattr(im_props[0], test_attribute), np.ndarray):                
            attributes_list += [test_attribute]
            
    return attributes_list
def regionprops_to_df(im_props):
    """
    Read content of all attributes for every item in a list
    output by skimage.measure.regionprops
    """

    attributes_list = scalar_attributes_list(im_props)

    # Initialise list of lists for parsed data
    parsed_data = []

    # Put data from im_props into list of lists
    for i, _ in enumerate(im_props):
        parsed_data += [[]]
        
        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(im_props[i], attributes_list[j])]

    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=attributes_list)

# %% Get DF from clusters
df = regionprops_to_df(clusters)
# %%