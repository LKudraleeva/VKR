import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage import io

from Model.TrainPart.metrics import *
from Model.TrainPart.utils import labeled_image_to_color


if __name__ == '__main__':
    model = load_model('../ModelSettings/Pigment-model.h5', custom_objects={'customized_loss': customized_loss,
                                                                            'dice_coefficient': dice_coefficient})

    testing_image = io.imread('../../Data/ResizedImages/0.png', as_gray=True)
    testing_image = testing_image.reshape((1, 256, 256, 1))
    prediction = model.predict(testing_image)
    prediction = np.squeeze(prediction, axis=0)
    color = labeled_image_to_color(prediction)
    plt.imshow(color, cmap='gray')
    plt.show()
