import numpy as np
from PIL import Image
from keras.models import load_model

from src.models.metrics import *
from src.data.preprocessing import preprocessing_images, labeled_image_to_color
import cv2

height, width = 256, 256


def predict(src: str, dest: str):
    images = preprocessing_images(src)
    p_model = load_model('../../models/Pigment_model2.h5', custom_objects={'customized_loss': customized_loss,
                                                                           'dice_coefficient': dice_coefficient})
    for i, image in enumerate(images):
        prediction = p_model.predict(image)
        color = labeled_image_to_color(np.squeeze(prediction, axis=0))
        processed_image = cv2.medianBlur(color, 5)
        img = Image.fromarray(processed_image, 'RGB')
        img.save(dest + str(i + 1) + '.png')


if __name__ == '__main__':
    predict('../../data/vmd/12/Images/', '../../data/vmd/12/Preds/')

