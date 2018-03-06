def VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(None, None, None), pooling=None, classes=1000):
    from keras.applications import vgg16
    from keras.layers import Input, Lambda
    import numpy as np


    def preprocess_input(x):
    	import numpy as np
    	return (x - np.array([123.68, 116.779, 103.939]))[:, :, :, ::-1]
    
    return vgg16.VGG16(include_top, weights,
                      input_tensor=Lambda(preprocess_input, name='preprocess')(Input(input_shape)),
                      pooling=pooling, classes=classes)
