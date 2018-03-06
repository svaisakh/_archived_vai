import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import traceback

from os.path import exists
from keras.models import load_model, model_from_json
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback
from vaidata import pickle_load, pickle_dump, get_images
from vaiutils import plot_images, concatenate_dict, summarize_tensor
from keras_tqdm import TQDMNotebookCallback
from tqdm import tqdm_notebook

def inflate_model(path, name, history_path=None, model=None):
    assert exists(path)

    if model is None:
        try:
            model = load_model(path)
        except:
            traceback.print_exc()
            print("There's an error. If you know the model architecture, try creating it, and passing it as an argument to this function. It will load the weights and history.")
            return
    else:
        model.load_weights(path)

    model.name = name
    if history_path is not None:    model.training_history = pickle_load(history_path, {})

    return model

def get_batches(directory, generator, batch_size=32, shuffle=True, target_size=(224, 224)):
    batches = generator.flow_from_directory(
        directory, target_size=target_size, batch_size=batch_size, shuffle=shuffle)
    batches.num_batches = int(batches.samples / batches.batch_size)
    return batches

def get_predictions(model, batches, loss, predictions=None, mode='random', num_images=9):
    assert mode in ['random', 'correct_random', 'incorrect_random', 'correct_confident', 'incorrect_confident', 'correct_doubtful', 'confused']

    if predictions is None:
        batches.reset()
        batches.total_batches_seen = 0
        predictions = model.predict_generator(batches, batches.num_batches, verbose=1)
    if type(loss) is np.ndarray:
        losses = np.copy(loss)
    else:
        losses = loss(predictions, batches.classes)
    filenames = np.array(batches.filenames)

    correct_idx = np.where(predictions.argmax(1) == batches.classes)[0]
    incorrect_idx = np.where(predictions.argmax(1) != batches.classes)[0]

    def intersection(list1, list2):
        return list(set(list(list1)).intersection(set(list(list2))))

    if mode == 'random':
        img_idx = np.random.randint(batches.samples, size=num_images)
    elif mode == 'correct_random':
        img_idx = correct_idx[np.random.randint(len(correct_idx), size=num_images)]
    elif mode == 'incorrect_random':
        img_idx = incorrect_idx[np.random.randint(len(incorrect_idx), size=num_images)]
    elif mode == 'correct_confident':
        img_idx = losses.argsort()[:num_images]
        img_idx = intersection(img_idx, correct_idx)
    elif mode == 'incorrect_confident':
        img_idx = losses.argsort()[-1:-num_images-1:-1]
        img_idx = intersection(img_idx, incorrect_idx)
    elif mode == 'correct_doubtful':
        img_idx = correct_idx[losses[correct_idx].argsort()[-1:-num_images-1:-1]]
    else:
        img_idx = predictions.max(1).argsort()[:num_images]

    return filenames[img_idx]

class LearningRateSetter(LearningRateScheduler):
    def __init__(self, epochs, lr):
        if type(epochs) is not list:
            epochs = [epochs]
            lr = [lr]
        assert len(lr) == len(epochs), "Length of epochs and lr must be same"

        self.lr_schedule = np.hstack([np.repeat(lr[i], epochs[i])
                                     for i in range(len(lr))])
        super().__init__(lambda epoch: self.lr_schedule[epoch])
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.schedule(epoch) is None:
            return
        super().on_epoch_begin(epoch, logs)

class ModelSave(ModelCheckpoint):
    def __init__(self, path, filename=None):
        if filename is None:
            self.path = path
            self.filename=None
            super().__init__(None, save_best_only=True)
            return

        self.filename=filename

        super().__init__(path + '/' + filename + '.h5', save_best_only=True)
        self.hist_path = path + '/' + filename + '-history.p'

    def set_model(self, model):
        super().set_model(model)
        if self.filename is None:
            self.__init__(self.path, self.model.name)
        if 'val_loss' in self.model.training_history.keys():
            if len(self.model.training_history['val_loss']) != 0:
                self.best = np.sort(np.array(self.model.training_history['val_loss']))[0] 
    
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if np.array(self.model.training_history['val_loss']).argsort()[0] == len(self.model.training_history['val_loss'])-1:
            pickle_dump(self.hist_path, self.model.training_history)

class UpdateHistoryCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        concatenate_dict(self.model.training_history, logs)

def fit_model(model, train_batches,  epochs=1, lr=None, val_batches=None, save_path='', callbacks=None):
    fit_callbacks = [TQDMNotebookCallback(), UpdateHistoryCallback(), LearningRateSetter(epochs, lr)]

    if save_path != '':
        fit_callbacks.append(ModelSave(save_path))

    if callbacks is not None:
        fit_callbacks.append(callbacks)

    if val_batches is None:
        model.fit_generator(train_batches, train_batches.num_batches,
                                                epochs=np.sum(epochs), verbose=0, callbacks=fit_callbacks)
    else:
        model.fit_generator(train_batches, train_batches.num_batches,
                                                epochs=np.sum(epochs), verbose=0, callbacks=fit_callbacks,
                                                validation_data=val_batches,
                                                validation_steps=val_batches.num_batches)

def summarize_model_weights(model, layer=None, idx=None, color='b'):
    if layer is None:
        for i, _ in enumerate(tqdm_notebook(model.layers)):
            summarize_model_weights(model, i, idx)
        return

    if idx is not None:
        if idx >= len(model.layers[layer].weights):
            return
        summarize_tensor(model.layers[layer].get_weights()[idx], model.layers[layer].weights[idx].name, color)
        plt.show()
    else:
        for weight in model.layers[layer].weights:
            summarize_tensor(K.eval(weight), weight.name, color)
            plt.show()

def reset_layer(model, layers):
    if type(layers) is not list:
        layers = [layers]

    new_model = model_from_json(model.to_json())

    for i, layer in enumerate(tqdm_notebook(new_model.layers)):
        if type(layers[0]) is int:
            if i not in layers:
                layer.set_weights(model.layers[i].get_weights())
        elif type(layers[0]) is str:
            if layer.name not in layers:
                layer.set_weights(model.layers[i].get_weights())

    return new_model