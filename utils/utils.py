'''import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from time import time
from tqdm import tqdm_notebook
from seaborn import kdeplot, rugplot'''

def rect_factors(number):
    square_root = int(np.sqrt(number))
    for divisor in range(square_root, number + 1):
        if divisor == 0:
            continue
        if number % divisor == 0:
            return (max(divisor, int(number / divisor)), min(divisor, int(number / divisor)))

def path_consts(create=False):
    from os import getcwd, makedirs
    from os.path import split, exists, expanduser, join
    from sys import path
    from glob import glob
    
    path_vars = {}
    DIR_DATA = path_vars['DIR_DATA'] = {}
    for p in  glob(expanduser('~/.data/*/' )):
        DIR_DATA[p.split('/')[-2]] = p[:-1]
    
    path_vars['DIR_NB'] = getcwd()
    path_vars['DIR_MAIN'] = split(getcwd())[0]
    path_vars['DIR_OUTPUT'] = join(path_vars['DIR_MAIN'], 'Outputs')
    path_vars['DIR_SCRIPTS'] = join(path_vars['DIR_MAIN'], 'scripts')
    path_vars['DIR_CHECKPOINTS'] = join(path_vars['DIR_MAIN'], 'Checkpoints')
    
    for p in path_vars.values():
        if type(p) is not str:
            continue
        if not exists(p) and create:
            makedirs(p, exist_ok=True)
            
    path.append(path_vars['DIR_MAIN'])
    path.append(path_vars['DIR_SCRIPTS'])
        
    return path_vars.items()
                
def handle_files(path_from, path_to, filenames, mode, verbose=True):
    assert mode in ['copy', 'move']
    from os.path import exists
    from glob import glob
    from shutil import copyfile
    from os import chdir, mkdir, rename

    chdir(path_from)
    if not exists(path_to):
        mkdir(path_to)

    if type(filenames) is not list:
        all_filenames = glob('*.jpg')

        num_files = int(filenames * len(all_filenames))
        filenames = np.random.choice(all_filenames, num_files, replace=False)

    if verbose:
        filename_iterator = tqdm_notebook(filenames)
    else:
        filename_iterator = filenames
    for filename in filename_iterator:
        if not exists(path_from + '/' + filename):
            continue

        if mode == 'copy':
            copyfile(path_from + '/' + filename, path_to + '/' + filename)
        else:
            rename(path_from + '/' + filename, path_to + '/' + filename)

def summarize_tensor(tensor, name='', color='b', rugs=False, kde=False):
    tensor = tensor.reshape(-1)

    if np.issubdtype(tensor.dtype, np.integer):
        text = "{}    {} <-> {} : [{}, {}]".format(name, tensor.mean(), tensor.std(), tensor.min(), tensor.max())
    else:
        text = "{}    {:1.2e} <-> {:1.2e} : [{:1.2e}, {:1.2e}]".format(name, tensor.mean(), tensor.std(), tensor.min(), tensor.max())

    if not kde:
        print(text)
        return

    if tensor.std() == 0:
        return
    kdeplot(tensor, shade=True, color=color)
    if rugs:
        if len(tensor) < 1e3:
            rugplot(tensor, color=color)
        else:
            rugplot(tensor[np.random.randint(0, len(tensor), int(1e3))], color=color)

    plt.title(text)

def randpick(array, size=1):
    from glob import glob

    if type(array) is str:
        array = glob(array)
        if len(array) == 0:
            return
    result = np.random.choice(array, size).tolist()
    return result if size != 1 else result[0]