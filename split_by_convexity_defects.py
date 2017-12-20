import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import skimage
from skimage import morphology, feature
from scipy import ndimage
from skimage.measure import label, regionprops
from mpl_toolkits.axes_grid1 import ImageGrid

def compactness(p):
    p = regionprops(label(p.filled_image))[0]
    return 4*np.pi*p.area/(p.perimeter**2)


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

def convexity_defects(im):
    ch = skimage.morphology.convex_hull_image(im).astype(int)
    d = ndimage.distance_transform_edt(ch)
    d[ndimage.binary_fill_holes(im)]=0
    peaks = skimage.feature.corner_peaks(d, indices=False, min_distance=20, threshold_abs=15)
    return peaks, d

def split_by_convdef(im):
    opr = regionprops(label(im))
    lpr = len(opr)
#    for p in pr:
#        if p.area<1000:
#            for x, y in p.coords:
#                im[x, y]=0
    if lpr>1:
        print('Already multiple nuclei, no need to split')
        return None
    
    if lpr==1 and opr[0].major_axis_length/opr[0].minor_axis_length<1.2:
        print('One nucleus')
        return None
    
    if lpr==0:
        print('No objects present')
        return None
    #Trying to split, if find big convexity defects
    #outim = im.copy()
    newim = skimage.filters.median(im, skimage.morphology.disk(25))
    newim = ndimage.binary_fill_holes(skimage.morphology.binary_closing(newim, skimage.morphology.disk(25)))
#    ch = skimage.morphology.convex_hull_image(newim).astype(int)
#    ch[newim]=-1
#    d = ndimage.distance_transform_edt(ch)
##    d[newim==1]=0
#    peaks = skimage.feature.corner_peaks(d, indices=False, min_distance=20, threshold_abs=15)
    peaks, d = convexity_defects(newim)
    pr = regionprops(label(peaks), intensity_image=d)
    if len(pr)>=2:
#        pr = regionprops(label(peaks), intensity_image=d)

        pr = sorted(pr, key=lambda x: x.mean_intensity, reverse=True)[:2]

        centres = [p.centroid for p in pr]
        assert len(centres)==2
        (x0, y0), (x1, y1) = centres
        length = int(np.hypot(x1-x0, y1-y0))
        xs = []
        ys = []
        for s1 in np.arange(-7, 8, 1):
            for s2 in np.arange(-7, 8, 1):
                xs.append(np.linspace(x0+s1, x1+s1, length, dtype=int))
                ys.append(np.linspace(y0+s2, y1+s2, length, dtype=int))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        return x, y
    if len(pr)<2 and opr[0].major_axis_length/opr[0].minor_axis_length>1.3:
        return False
    else:
        return None

def apply_convdef(im):
    im = im.astype(bool)
    newim = np.zeros_like(im).astype(bool)
    zs = np.array([])
    xs = np.array([])
    ys = np.array([])
    for i, image in enumerate(im):
        if np.any(image):
            m = split_by_convdef(image)
            if m is None:
                print(i, 'No convexivity defects detected')
                newim[i]=image
            elif m is False:
                print(i, 'No defects, but elongated - removing')
                newim[i]=np.zeros_like(image)
            else:
                print('Splitting', i)
                x, y = m
                for j in range(max([0, i-3]), min([i+4, im.shape[0]])):
                    newim[j] = image
                    xs = np.concatenate([xs, x])
                    ys = np.concatenate([ys, y])
                    zs = np.concatenate([zs, [j]*len(x)])
    if len(xs)>0:
        newim[zs.astype(int), xs.astype(int), ys.astype(int)]=False
    for i, image in enumerate(newim):
        image=skimage.morphology.remove_small_objects(image.astype(bool), 5000)
        pr = regionprops(label(image, connectivity=1))
        for p in pr: 
            if compactness(p)<0.35:
                for j1, j2 in p.coords:
                    image[j1, j2] = False
        newim[i] = image
    newim *= im #Make sure we didn't fill any hole anywhere
    
    ### Dilating holes
    s = np.zeros((21, 21))
    s[10, 10] = 1
    struct = np.array([s, s, skimage.morphology.disk(10), s, s])
    antiregions = skimage.segmentation.clear_border(~newim)
    antiregions = ndimage.binary_dilation(antiregions, structure=struct)
    newim *= ~antiregions
    
    ### Eroding to remove nuclear periphery and perinucleolar heterochromatin
    s = np.zeros((21, 21))
    s[10, 10] = 1
    struct = np.array([s, s, skimage.morphology.disk(20), s, s])
    newim = skimage.morphology.binary_erosion(newim, struct)
    
    newim = label(newim, connectivity=1)
    newim = skimage.segmentation.relabel_sequential(skimage.morphology.remove_small_objects(newim, 300000))[0]
    return newim

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    args = parser.parse_args()
    imfile = args.imagefile
    image = tf.TiffFile(imfile).asarray().squeeze()[:, 0]
    name = name = imfile.split('/')[-1]
    im = np.load('/exports/eddie/scratch/s1529682/image/segmentations/nows/%s.npy' % name)
    im = apply_convdef(im)
    
#    im = np.load('/exports/eddie/scratch/s1529682/image/segmentations/conv/%s.npy' % name)
        
    plot_segm(image, '/exports/eddie/scratch/s1529682/image/segmentimage/conv/', im, name=name)
    np.save('/exports/eddie/scratch/s1529682/image/segmentations/conv/'+name, im)
