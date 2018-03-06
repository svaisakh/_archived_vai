import numpy as np
import itertools
import matplotlib.pyplot as plt

from skimage.transform import resize
from vai_.utils.utils import rect_factors
from scipy.signal import savgol_filter

def plot_images(images, titles=None, pixel_range=(0, 255), cmap=None, merge_shape=None, resize='smin', retain=False, savepath=None):
    if type(images) is str:
        from glob import glob
        images = [plt.imread(f) for f in glob(images, recursive=True)]
        assert len(images) != 0, "File not found!"

    if type(images) == np.ndarray:
        images = list(images)
        
    images = __resize_images(__colorize_images(images), resize)
    if titles == '':
        titles = [''] * len(images)
    if titles is None:
        __show_image(__merge_images(images, merge_shape), '', pixel_range, cmap)
    else:
        if merge_shape is None:
            merge_shape = rect_factors(len(images))[::-1]

        fig, axes = plt.subplots(merge_shape[0], merge_shape[1])
        for i, ax in enumerate(axes.flat):
            __show_image(images[i], titles[i], pixel_range, cmap, ax)
    
    if savepath is not None:
        plt.savefig(savepath, dpi=400, bbox_inches='tight')
    if not retain:
        plt.show()

def smooth_plot(signal, frac_smooth=0.3, remove_outlier=True, keys=None, title=''):
    if type(signal) is dict:
        for title, signal_val in signal.items():
            if keys is None:
                smooth_plot(signal_val, frac_smooth, remove_outlier, title=title)
            elif title in keys:
                smooth_plot(signal_val, frac_smooth, remove_outlier, title=title)
        plt.legend()
        return
                    
    plt.plot(__smoothen(signal, 0, remove_outlier), '.-', alpha=0.3, label=title)
    plt.plot(__smoothen(signal, frac_smooth, remove_outlier), label=title)

def __merge_images(images, shape=None):
    if shape is None:
        shape = rect_factors(len(images))[::-1]
    elif shape is 'row':
        shape = (1, len(images))
    elif shape is 'column':
        shape = (len(images), 1)
        
    assert np.prod(shape) == len(images), 'Improper merge shape'
    assert all(np.std([i.shape for i in images], 0) == 0), 'All images need to be the same shape'
    
    img_shape = np.array(images[0].shape[:-1])
    merged_image = np.zeros(np.append(img_shape * np.array(shape), 3))
    
    for idx, (row, column) in enumerate(list(itertools.product(range(shape[0]), range(shape[1])))):
        merged_image[row*img_shape[0]:(row + 1)*img_shape[0],
                    column*img_shape[1]:(column + 1)*img_shape[1], :] = images[idx]
    
    return merged_image

def __resize_images(images, img_shape='smin'):
    if img_shape is None:
        return images
    
    if type(img_shape) is not tuple and type(img_shape) is not list:
        shapes = np.array([image.shape[:-1] for image in images])
        if np.all(shapes.std(0) == 0):
            return images
        
        if img_shape[0] == 's':
            shapes = np.array([[int(np.sqrt(np.prod(s)))]*2 for s in shapes])
            img_shape = img_shape[1:]
            
        if img_shape == 'min':
            img_shape = shapes.min(0)
        elif img_shape == 'max':
            img_shape = shapes.max(0)
        elif img_shape == 'mean':
            img_shape = shapes.mean(0)
        else:
            assert False, "'img_shape' must be one of 'min', 'max' or 'mean' or the desired shape"

        img_shape = img_shape.astype(int)
    
    return [(resize(image, img_shape, mode='constant') * 255).astype(np.uint8) for image in images]

def __show_image(img, title=None, pixel_range=(0, 255), cmap=None, ax=None):
    if pixel_range == 'auto':
        pixel_range = (images.min(), images.max())
    if ax is None:
        plt.imshow(((img-pixel_range[0])/(pixel_range[1]-pixel_range[0])), cmap, vmin=0, vmax=1)
        plt.title(title)
        plt.xticks([]); plt.yticks([])
    else:
        ax.imshow(((img-pixel_range[0])/(pixel_range[1]-pixel_range[0])), cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

def __colorize_images(images):
    color_images = []
    for image in images:
        assert len(image.shape) == 2 or len(image.shape) == 3, 'Incorrect image dimensions'
        if len(image.shape) == 2:
            color_images.append(np.repeat(np.expand_dims(images, -1), 3, -1))
        else:
            assert image.shape[-1] == 3 or image.shape[-1] == 1, 'Incorrect image dimensions'
            if image.shape[-1] == 3:
                color_images.append(image)
            else:
                color_images.append(np.repeat(images, 3, -1))
                
    return color_images

def __smoothen(signal, frac_smooth=0.3, remove_outlier=True):
    if type(signal) != list and type(signal) != np.ndarray:
        from copy import copy as shallow_copy
        signal_line = signal.axes.lines[0]
        smooth_line = shallow_copy(signal_line)
        signal_x, signal_y = signal_line.get_data()
        smooth_y = __smoothen(signal_y, frac_smooth, False)
        smooth_line.set_data(signal_x, smooth_y)
        smooth_line.set_color('g')
        signal_line.set_alpha(0.2)
        signal.axes.add_line(smooth_line)
        return signal

    def __median_absolute_deviation_outlier(points, thresh=3.5):
            if len(points.shape) == 1:
                points = points[:,None]
            median = np.median(points, axis=0)
            diff = np.sum((points - median)**2, axis=-1)
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)

            modified_z_score = 0.6745 * diff / med_abs_deviation

            return modified_z_score > thresh
        
    x = np.array(signal)
    if remove_outlier:
        outliers = __median_absolute_deviation_outlier(x)
        x=x[~outliers]
    window_length = int(x.shape[0] * frac_smooth)
    if window_length % 2 == 0:
        window_length += 1

    if window_length < 3:
        return x
    elif window_length > x.shape[0]:
        window_length = x.shape[0]
        if window_length % 2 == 0:
            window_length -= 1

    return savgol_filter(x, window_length, 1)
