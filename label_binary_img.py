# %%
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import color, io, measure
from scipy import ndimage
import pandas as pd

im = cv2.imread('pos15.jpg', 0)

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
msk_circ = create_circular_mask(im.shape[0], im.shape[1])
ret, th = cv2.threshold(im[msk_circ], 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
im_th = im.copy()
im_th[im<ret] = 0
im_th[im>ret] = 255
eroded = cv2.erode(im_th, np.ones((5,5), np.uint8), iterations = 3)
dilated = cv2.dilate(eroded, np.ones((5,5), np.uint8), iterations = 3)
plt.imshow(dilated )

# %%
# Labeling individual regions
mask = dilated> ret
# Define what kind of connectivity matrix to use. This is 8-connected matrix
s = [[1,1,1],[1,1,1],[1,1,1]]

lbl_msk, num_lbl = ndimage.label(mask, structure=s)

img2 = color.label2rgb(lbl_msk, bg_label =0)
plt.imshow(img2)
# %%
#Analyze clusters
clusters = measure.regionprops(lbl_msk, im)
#%%

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


# %%
df = regionprops_to_df(clusters)
# %%