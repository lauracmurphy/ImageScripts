import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['pdf.fonttype'] = 42
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import skimage
from scipy import ndimage
from skimage.filters import threshold_otsu, gaussian, median
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_segm(image, loc, regions=None, name=None):
    if name is None:
        name=''
    l = im.shape[0]//8+1
    fig = plt.figure(figsize=(12, l*1.5))
    axarr = ImageGrid(fig, 111,
                 nrows_ncols=(l, 8),
                 axes_pad=0.01,
                 )
    for i in range(len(image)):
        axarr[i].imshow(image[i], cmap='Greys_r', vmax=image.max())
        if regions is not None and not np.all(regions[i]==0):
            axarr[i].imshow(np.ma.masked_equal(regions[i], 0), vmin=0, vmax=regions.max(), cmap='viridis', alpha=0.15)
        axarr[i].axis('off')
        axarr[i].text(0.0, 1.0, i,
             horizontalalignment='left',
             verticalalignment='top',
             transform = axarr[i].transAxes, color='w')
    for j in range(i, len(axarr)):
        axarr[j].axis('off')
    plt.savefig(loc+name+'.png',
                 bbox_inches='tight', dpi=300)
    plt.close()
    
def compactness(p):
    p = regionprops(label(p.filled_image))[0]
    return 4*np.pi*p.area/(p.perimeter**2)

def segment_z(subim):
    if np.all(subim==0):
        return subim

    a = skimage.restoration.denoise_tv_chambolle(subim, multichannel=False, weight=0.2)

    mask = a>threshold_otsu(a)
    mask = clear_border(mask)
    mask = skimage.morphology.binary_closing(mask, selem=skimage.morphology.disk(3))
    mask = skimage.morphology.remove_small_holes(mask, 50)
    mask = skimage.morphology.remove_small_objects(mask, 1000)
    antiregions = label(clear_border(~mask))
    pr = regionprops(antiregions)
    for p in pr:
        #Filter holes
        if p.major_axis_length/p.minor_axis_length>1.25 or compactness(p)<0.5:
            for j1, j2 in p.coords:
                mask[j1, j2] = True
    newmask = skimage.morphology.remove_small_objects(ndimage.binary_fill_holes(mask).astype(bool), 15000)
    newmask = skimage.filters.median(newmask, selem=skimage.morphology.disk(20))
           
    pr = regionprops(label(newmask), subim)
    for p in pr:
        try:
            if compactness(p)<0.35 or p.area>150000 or \
               p.mean_intensity<1.5*subim.mean():
                for j1, j2 in p.coords:
                    newmask[j1, j2] = False
        except TypeError:
            for j1, j2 in p.coords:
                newmask[j1, j2] = False
    newmask *= mask
    return newmask

def segment_image(image):
    im = np.clip(image, 0, np.percentile(image, 99)).astype(int)
    print('Thresholding z planes')
    mask = np.zeros_like(im)
    for i, subim in enumerate(im):
        mask[i] = segment_z(subim)
    mask = skimage.morphology.remove_small_objects(mask.astype(bool), 300000)

    regions = label(mask, connectivity=1)
    
    np.save('/exports/eddie/scratch/s1529682/image/segmentations/nows/'+name, regions)
    

    plot_segm(image, '/exports/eddie/scratch/s1529682/image/segmentimage/nows/', regions, name=name)


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    args = parser.parse_args()
    imagefile = args.imagefile
    
    name = imagefile.split('/')[-1]
    print(name)
    with tf.TiffFile(imagefile) as f:
        im = f.asarray().squeeze()[:, 0]
    im = skimage.exposure.rescale_intensity(im, out_range=(0, 255))
    segment_image(im)   
    print('Done')
