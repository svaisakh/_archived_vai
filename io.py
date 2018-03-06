import numpy as np
from os.path import exists
from scipy.misc import imread, imresize

def pickle_load(filename, default=None, has_lambda=False):
    if not exists(filename):
        return default

    if has_lambda: import dill as pickle
    else: import pickle

    with open(filename, 'rb') as f:
        return pickle.load(f)

def pickle_dump(filename, obj, has_lambda=False):
    if has_lambda: import dill as pickle
    else: import pickle
    
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def get_images(filenames, size, path=None):
    images = np.zeros((len(filenames), size[0], size[1], size[2]))
    for i, filename in enumerate(filenames):
        if path is None:
            images[i] = imresize(imread(filename), (size[0], size[1]))
        else:
            images[i] = imresize(imread(path + '/' + filename), (size[0], size[1]))
    return images