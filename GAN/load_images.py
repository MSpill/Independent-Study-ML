from PIL import Image
from pathlib import Path
import pickle
import numpy as np


def load(num=-1):
    train_images = []
    i = 0

    for path in Path('data/small_images').rglob('*.jpg'):
        print(path)
        image = Image.open(path)
        # need to put the image in the right orientation and normalize data values to between -1 and 1
        train_images.append(np.asarray(image).transpose(1,0,2) * 1.0 / 127 - 1.0)
        i += 1
        if num != -1 and i >= num:
            break

    train_images = np.asarray(train_images).astype('float32')
    return train_images