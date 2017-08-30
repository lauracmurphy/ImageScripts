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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("imagefile")
args = parser.parse_args()
imagefile = args.imagefile

def plot_segm(image, loc, regions=None, name=None):
    if regions is None:
        regions=np.zeros_like(image)
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
        ri = regions[i]
#        edge = np.ma.masked_equal(ndimage.filters.maximum_filter(ri, size=5)-ri, 0)
        axarr[i].imshow(np.ma.masked_equal(ri, 0), vmin=0, vmax=regions.max(), cmap='viridis', alpha=0.15)
        axarr[i].axis('off')
        axarr[i].text(0.0, 1.0, i,
             horizontalalignment='left',
             verticalalignment='top',
             transform = axarr[i].transAxes, color='w')
    for j in range(i, len(axarr)):
        axarr[j].axis('off')
#    plt.savefig('/exports/igmm/eddie/wendy-lab/ilia/output/image/segmentimage/nows/'+name+'.png',
#                bbox_inches='tight', dpi=300)
    plt.savefig(loc+name+'.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
def compactness(p):
    p = regionprops(label(p.filled_image))[0]
    return 4*np.pi*p.area/(p.perimeter**2)

def not_fused(p):
    return p.area>100000 and p.major_axis_length/p.minor_axis_length<1.5

def apply_randomwalker2D(image):
    distance = ndimage.distance_transform_edt(image)
    distance = np.array(distance)
    local_maxi = peak_local_max(distance, min_distance=100,
                                num_peaks_per_label=2,
                                indices=False, labels=image)
    markers = label(local_maxi)
    markers[image==0] = -1
    labels_rw = skimage.segmentation.random_walker(image, markers, mode='bf')
    labels_rw[labels_rw==-1]=0
    return labels_rw

def apply_randomwalker3D(image, spacing):
    for i, z in enumerate(image):
        image[i] = ndimage.binary_fill_holes(z)
    distance = ndimage.distance_transform_edt(image, sampling=spacing)
    distance = ndimage.maximum_filter(distance, footprint=np.ones((10, 50, 50)))
    local_maxi = peak_local_max(distance, num_peaks_per_label=2,
                                indices=False)
    markers = label(local_maxi)
    markers[image==0] = -1
    labels_rw = skimage.segmentation.random_walker(image, markers,
                                                   spacing=spacing,
                                                   mode='cg_mg')
    labels_rw[labels_rw==-1]=0
    return labels_rw

def apply_watershed(image):
    distance = []
    for i, z in enumerate(image):
        distance.append(ndimage.distance_transform_edt(ndimage.binary_fill_holes(z)))
    distance = np.array(distance)
    local_maxi = peak_local_max(distance, num_peaks_per_label=2,
                                indices=False, footprint=np.ones((30, 100, 100)),
                                labels=image)
    markers = skimage.measure.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=image,
                          watershed_line=True)
    return labels_ws

#def apply_watershed3Ddistance(image, sampling):
#    for i, z in enumerate(image):
#        image[i] = ndimage.binary_fill_holes(z)
#    distance = ndimage.distance_transform_edt(image, sampling=sampling)
#    local_maxi = peak_local_max(distance, num_peaks_per_label=2,
#                                indices=False, footprint=np.ones((30, 100, 100)),
#                                labels=image)
#    markers = skimage.measure.label(local_maxi)
#    labels_ws = watershed(-distance, markers, mask=image,
#                          watershed_line=True)
#    return labels_ws
    
def apply_watershed3Ddistance(image, sampling):
    for i, z in enumerate(image):
        image[i] = ndimage.binary_fill_holes(z)
    distance = ndimage.distance_transform_edt(image, sampling=spacing)
    distance = ndimage.maximum_filter(distance, footprint=np.ones((10, 50, 50)))
    local_maxi = peak_local_max(distance, num_peaks_per_label=2,
                                indices=False)
    markers = label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=image,
                          watershed_line=True)
    return labels_ws

def segment_z(subim):
    if np.all(subim==0):
        return subim

    a = gaussian(median(subim, np.ones((5, 5))), 5)

    mask = a>threshold_otsu(a)
    mask = skimage.morphology.binary_closing(mask, selem=np.ones((10, 10)))
    mask = skimage.morphology.remove_small_holes(mask, 50)

    antiregions = label(clear_border(~mask))
#    antiregions = skimage.morphology.remove_small_objects(antiregions, 50)
    pr = regionprops(antiregions)
    for p in pr:
        #Filter holes
        if p.area<50 or p.major_axis_length/p.minor_axis_length>1.5 or compactness(p)<0.5:
            for j1, j2 in p.coords:
                mask[j1, j2] = True
                
    mask = skimage.morphology.remove_small_holes(mask, 500)
    mask = skimage.morphology.opening(mask, np.ones((10, 10)))
    mask = skimage.morphology.remove_small_objects(mask, 10000)
    ###Randomwalker
#    regions = apply_randomwalker2D(ndimage.binary_fill_holes(mask))
#    regions[~mask]=0
#    regions = skimage.morphology.erosion(regions, skimage.morphology.diamond(6))
    
#    regions = skimage.segmentation.relabel_sequential(skimage.morphology.remove_small_objects(regions, 1000))[0]
    ###
           
    pr = regionprops(label(mask), subim)
    for p in pr:           
        if compactness(p)<0.35 or p.area>300000 or p.mean_intensity>5*subim.mean():
            for j1, j2 in p.coords:
                mask[j1, j2] = False
    return mask

def segment_image(im, spacing=1):
    im = np.clip(im, 0, np.percentile(im, 99)).astype(int)
    mask = []
    for subim in im:
        mask.append(segment_z(subim))
    mask = np.array(mask)
    mask = skimage.morphology.remove_small_objects(mask, 300000)
#    mask = np.array(mask)>0
    
#    regions = skimage.morphology.remove_small_holes(regions, 1000)
#    regions = skimage.morphology.erosion(regions, np.ones((3, 12, 12)))
    regions = label(mask)
    
    regions_ws = apply_watershed(mask)
    regions_ws[~mask]=0
    
    regions_ws3d = apply_watershed3Ddistance(mask, spacing)
    regions_ws3d[~mask]=0
    
    regions_rw3d = apply_randomwalker3D(mask, spacing)
    regions_rw3d[~mask]=0
    
    return regions, regions_ws, regions_ws3d, regions_rw3d

if __name__=='__main__':
    name = imagefile.split('/')[-1]
    print(name)
    with tf.TiffFile(imagefile) as f:
        for line in f.info().split('\n'):
            if 'voxel_size_x' in line:
                xres = float(line.split()[-1])
            if 'voxel_size_y' in line:
                yres = float(line.split()[-1])
            if 'voxel_size_z' in line:
                zres = float(line.split()[-1])
        im = f.asarray().squeeze()[:, 0]
    spacing = zres, xres, yres
    im = skimage.exposure.rescale_intensity(im, out_range=(0, 256))
    regions, regions_ws, regions_ws3d, regions_rw3d = segment_image(im, spacing=spacing)
    
    np.save('../../output/image/segmentations/nows/'+name, regions)
    np.save('../../output/image/segmentations/ws/'+name, regions_ws)
    np.save('../../output/image/segmentations/ws3d/'+name, regions_ws3d)
    np.save('../../output/image/segmentations/rw/'+name, regions_rw3d)
    
    plot_segm(im, '../../output/image/segmentimage/nows/', regions, name)
    plot_segm(im, '../../output/image/segmentimage/ws/', regions_ws, name)
    plot_segm(im, '../../output/image/segmentimage/ws3d/', regions_ws3d, name)
    plot_segm(im, '../../output/image/segmentimage/rw/', regions_rw3d, name)
    
    print('Done')
