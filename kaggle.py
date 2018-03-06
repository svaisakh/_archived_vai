import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

def plot_prediction_density(model, batches):
    plt.hist(model.predict_generator(batches, batches.num_batches).argmax(1), 
            normed=True, bins=np.arange(0, 11) - 0.5, label='Predicted')
    plt.hist(batches.classes, color='r', alpha=0.2, 
            normed=True, bins=np.arange(0, 11) - 0.5, label='Actual')
    plt.xlabel('Classes')
    plt.ylabel('Frequency of prediction/occurance')
    plt.legend()
    plt.title('Class frequencies')
    plt.show()
    
def plot_confusion_matrix(model, batches):
    conf_matrix = confusion_matrix(model.predict_generator(batches, batches.num_batches).argmax(1), batches.classes)
    cm = conf_matrix.copy()
    cm[range(len(cm)), range(len(cm))] = 0
    plt.imshow(cm, cmap='seismic')
    plt.grid(False)
    plt.xlabel = 'Actual'
    plt.ylabel = 'Predicted'
    plt.title('Confusion Matrix')
    plt.show()
    return conf_matrix
    
def dataset_consts(DIR_DATA, use_sample=True):
    dataset_consts = {}
    if use_sample:
        dataset_consts['DIR_TRAIN'] = DIR_DATA + '/sample/train'
        dataset_consts['DIR_VALID'] = DIR_DATA + '/sample/valid'
        dataset_consts['DIR_TEST'] = DIR_DATA + '/sample/test'
    else:
        dataset_consts['DIR_TRAIN'] = DIR_DATA + '/train'
        dataset_consts['DIR_VALID'] = DIR_DATA + '/valid'
        dataset_consts['DIR_TEST'] = DIR_DATA + '/test'
        
    return dataset_consts.items()