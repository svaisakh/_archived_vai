import torch

cuda = lambda x: x.cuda() if torch.cuda.is_available() else x

def channels_last(images):
    if len(images.shape) == 4:
        return images.transpose((0, 2, 3, 1))
    else:
        return images.transpose((1, 2, 0))
