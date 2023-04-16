import os

import numpy as np
from PIL import Image
from keras.models import load_model
from skimage import io

from model.training.metrics import *
from model.training.utils import labeled_image_to_color

if __name__ == '__main__':

    path_name = '../../data/vmd/4/Resize/'
    files = sorted(os.listdir(path_name), key=lambda file: int(file.split('.')[0]))
    filepath = [path_name + f for f in files]

    p_model = load_model('../saved_models/Pigment_model2.h5', custom_objects={'customized_loss': customized_loss,
                                                                              'dice_coefficient': dice_coefficient})
    for i, f in enumerate(filepath):
        test_image = io.imread(f, as_gray=True)
        test_image = test_image.reshape((1, 256, 256, 1))
        prediction = p_model.predict(test_image)
        color = labeled_image_to_color(np.squeeze(prediction, axis=0))
        img = Image.fromarray(color, 'RGB')
        img.save('../../data/vmd/4/Preds/' + str(i+1) + '.png')
