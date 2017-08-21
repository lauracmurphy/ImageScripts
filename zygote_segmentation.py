import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile as tf
from skimage.feature import greycomatrix, greycoprops
import skimage
from skimage import data
from scipy import ndimage
from skimage.filters import threshold_otsu, gaussian, median
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import glob
from scipy import stats
import scipy
import seaborn as sns
import matplotlib as mpl

def plot_segm(image, regions, name):
    f, axarr = plt.subplots(len(image), 2, figsize=(4, 2*len(image)))
    for i in range(len(image)):
        axarr[i, 0].imshow(image[i])
        axarr[i, 1].imshow(regions[i], vmax=regions.max())
        axarr[i, 0].axis('off')
        axarr[i, 1].axis('off')
        axarr[i, 0].text(0.0, 1.0, i,
             horizontalalignment='left',
             verticalalignment='top',
             transform = axarr[i, 0].transAxes, color='w')
        plt.subplots_adjust(wspace=0.1, hspace=0)#, bottom=None, top=None, left=None, right=None)
#         plt.tight_layout()
    plt.savefig('/home/ilya/Documents/biology/Vienna single-cell Hi-C/images/segmentation/'+name+'.png', bbox_inches='tight')
    plt.close()
   
def compactness(p):
    p = regionprops(label(p.filled_image))[0]
    return 4*np.pi*p.area/(p.perimeter**2)

def segment_image1(im):
#     if im.dtype!=np.uint8:
#         im = skimage.exposure.rescale_intensity(im, 'dtype', 'uint8')
    regions_by_slice = []
    for i in range(len(im)):
        subim = im[i]
        a = gaussian(median(subim, np.ones((5, 5))), 5)
        if np.all(a==0):
            regions_by_slice.append(a)
            continue
        mask = a>skimage.filters.threshold_otsu(a)
        mask = skimage.morphology.binary_closing(mask)
        mask = skimage.morphology.remove_small_holes(mask, 50)

        antiregions = label(skimage.segmentation.clear_border(~mask))
#         antiregions = skimage.morphology.remove_small_objects(antiregions, 50)
        pr = regionprops(antiregions)
        for p in pr:
            #Filter holes
            if p.major_axis_length/p.minor_axis_length>1.5 or compactness(p)<0.5:
                for j1, j2 in p.coords:
                    mask[j1, j2] = True
        regions = label(mask)
        regions = skimage.morphology.remove_small_objects(regions, 5000)
        
#         regions = skimage.morphology.remove_small_holes(regions, 500)
#         for j in set(list(regions.flatten())):
#             if np.sum(regions==j)>400000:
#                 regions[regions==j] = 0
        
        regions=label(regions)
        
        pr = regionprops(regions, subim)
        for p in pr:
            #Filter regions
            if p.area>100000 or compactness(p)<0.15 or p.mean_intensity/subim.mean()<2:
                for j1, j2 in p.coords:
                    regions[j1, j2] = False
#         regions = label(skimage.morphology.binary_closing(regions))
        regions_by_slice.append(regions)

    regions = np.array(regions_by_slice)
    mask = regions>0
    del regions_by_slice
    regions = label(skimage.morphology.binary_closing(regions))
    regions = skimage.morphology.remove_small_objects(regions, 100000)
    regions = skimage.morphology.remove_small_holes(regions, 500)
    return label(regions)
    
