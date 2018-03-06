import numpy as np
from keras import backend as K
from tqdm import tqdm_notebook

def dream(model, images=None, filter_idx=(1,0), num_iterations=20, strength=1, flags=[]):
    layer_num, filter_index = filter_idx
    
    input_img = model.input

    layer_output = model.layers[layer_num].output
    if filter_index < 0:
        loss = K.mean(layer_output)
    elif len(layer_output.shape) == 4:
        loss = K.mean(layer_output[:, :, :, filter_index])
    else:
        loss = K.mean(layer_output[:, filter_index])

    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-10)

    iterate = K.function([input_img, K.learning_phase()], [loss, grads])
    
    if images is None:
        image_shape = [1]+list(model.input_shape[1:])
        if image_shape[1] is None:
            image_shape = [1, 256, 256, 3]
        images = np.random.random(image_shape) * 20 + 128
    elif len(images.shape) == 3:
        images = np.stack([images])

    if 'verbose' in flags:
        it_range=tqdm_notebook(range(num_iterations))
    else:
        it_range=range(num_iterations)
        
    def deprocess_image(x):
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        x+= 0.5
        x = np.clip(x, 0, 1)

        x*= 255
        x = np.clip(x, 0, 255)
        return x
        
    for _ in it_range:
        loss_value, grads_value = iterate([images])
        images += grads_value * strength
        images = deprocess_image(images)
        if 'verbose' in flags:
            print("Loss: {}, Mean gradient: {}".format(loss_value, grads_value.mean()), end='\r', flush=True)
    
    if images.shape[0] == 1:
        return deprocess_image(images)[0]
    else:
        return deprocess_image(images)[0]
