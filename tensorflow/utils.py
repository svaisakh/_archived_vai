def read_mnist_data():
    data = read_data_sets("data/MNIST/", one_hot=True)
    
    print("Data shapes:")
    print("Training - x: {}, y:{}".format(data.train.images.shape, data.train.labels.shape))
    print("Test - x: {}, y:{}".format(data.test.images.shape, data.test.labels.shape))
    print("Validation - x: {}, y:{}".format(data.validation.images.shape, data.validation.labels.shape))
    
    data.test.cls = np.argmax(data.test.labels, axis=1)
    data.validation.cls = np.argmax(data.validation.labels, axis=1)
    
    params = {}
    img_size =params['img_size'] = int(np.sqrt(data.train.images.shape[1]))
    params['img_size_flat'] = img_size**2
    params['img_shape'] = (img_size, img_size)
    params['num_classes'] = data.train.labels.shape[1]
    params['num_channels'] = 1
    
    
    return data, params

def add_variable_histogram_summaries():
    v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(var.name, var)
        
    tf.logging.set_verbosity(v)
