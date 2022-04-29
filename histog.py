#%%
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.restoration import denoise_nl_means, estimate_sigma
# %%
im = img_as_float(io.imread('pos15.jpg'))
im_sc = rescale(im, scale = 0.25, anti_aliasing=True)
# %%
plt.imshow(im_sc)
# %%
plt.hist(im_sc.flat, bins = 100, range=(0,1))
# %%
patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)

sigma_est = np.mean(estimate_sigma(im, multichannel=True))
im_noise = denoise_nl_means(im_sc, h=1.15 * sigma_est, fast_mode=True,
                           **patch_kw)
plt.hist(im_noise.flat, bins = 100, range=(0,1))
# %%
from skimage import exposure
im_eq = exposure.equalize_adapthist(im_noise)
plt.hist(im_eq.flat, bins = 100, range=(0,1))
# %%
plt.imshow(im_eq)

# %%
im_lbl = im_eq.copy()
med = 0.7
im_lbl[(im_lbl> med -0.05) & (im_lbl< med + 0.05)] = 1
plt.imshow(im_lbl)
# %%
marker = np.zeros(im_eq.shape, dtype = np.uint)
marker[(im_lbl> 0.65) & (im_lbl< 0.75)] = 1
marker[(im_lbl< 0.65)] = 2

from skimage.segmentation import random_walker
labels = random_walker(im_eq, marker, beta = 200, mode = 'bf')

# %%
final = np.zeros((im_eq.shape[0], im_eq.shape[1], 3))
final[labels == 2] = (0,0,0)
final[labels == 1] = (1,0,0)
plt.imshow(final)
# %%
