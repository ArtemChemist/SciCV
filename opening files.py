# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, restoration
from skimage.filters.rank import entropy
from skimage.morphology import disk

# %%
img = io.imread('cfu.jpg', as_gray=True)
plt.hist(img, 100, (0,1))
# %%
from skimage.filters import threshold_otsu, try_all_threshold
from skimage.feature import canny
import glob

path = './Selected/*'
# %%
for file in glob.glob(path):
    im = io.imread(file, as_gray=True)
    thr = canny(im, sigma = 1, low_threshold=0.2, high_threshold=0.3)
    im[thr]=1
    im[~thr]=0
    file_save = '.'+file.split('.')[1]+'_can.jpg'
    io.imsave(file_save, im)
# %%
import scipy.ndimage as nd

# %%
gaussian_im =  nd.gaussian_filter(img, sigma = 10)
plt.imsave('gaussin.jpg',gaussian_im)

# %%
from skimage.restoration import denoise_nl_means, estimate_sigma
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
nlm = denoise_nl_means(img, h = 1.15 *sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)
plt.imsave('nlm.jpg', nlm)
# %%
