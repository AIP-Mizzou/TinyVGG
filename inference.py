import tensorflow as tf
import numpy as np
import pandas as pd
import glob
from PIL import Image
import json

labels = {
    0: 'lifeboat',
    1: 'ladybug',
    2: 'pizza',
    3: 'bell pepper',
    4: 'school bus', 
    5: 'koala',
    6: 'espresso',
    7: 'red panda',
    8: 'orange',
    9: 'sports car'
}

def main():
    path = './VOC/'
    ext = '*.jpg'
    files = glob.glob(path + ext)
    files.sort()

    tiny_vgg = tf.keras.models.load_model('trained_vgg_best.h5')
    tiny_vgg.summary()

    for path in files:
        data = np.zeros((1, 64, 64, 3))

        image = np.asarray(Image.open(path)).resize(64,64)
        data[0,:,:,:] = image

        result = tiny_vgg.predict(data).argmax()
        article = 'an' if labels[result][0] in 'aeiou' else 'a'
        print(f'Image {path} is {article} {labels[result]}.')

if __name__ == '__main__':
    main()