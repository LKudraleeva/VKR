import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import os

from matplotlib import pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split

from src.data.preprocessing import labels_encoding
from src.models.metrics import *
from src.models.model import get_model

EPOCHS = 100
BATCH_SIZE = 20


def get_dataset(images_path, labels_path):
    images = [io.imread(images_path + im, as_gray=True) for im in os.listdir(path=images_path)]
    labels = [io.imread(labels_path + l, as_gray=True) for l in os.listdir(path=labels_path)]

    images = np.array(images)
    labels = np.array(labels)

    train_images = images.reshape((images.shape[0], 256, 256, 1))
    train_labels = labels_encoding(np.asarray(labels))

    train_images = train_images.astype('float32')
    train_labels = train_labels.astype('float32')

    return train_test_split(train_images, train_labels, test_size=0.1, random_state=42)


def save_reports(history):
    epochs = range(10, 81, 2)

    plt.plot(epochs, history.history['dice_coefficient'][9::2], 'bo', label='Training dice_coefficient')
    plt.plot(epochs, history.history['val_dice_coefficient'][9::2], 'b', label='Validation dice_coefficient')
    plt.legend()
    plt.ylabel('dice_coefficient')
    plt.xlabel('epoch')
    plt.savefig('reports/dice.png')

    plt.plot(epochs, history.history['loss'][9::2], 'bo', label='Training loss')
    plt.plot(epochs, history.history['val_loss'][9::2], 'b', label='Validation loss')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('reports/loss.png')


def train_model():
    model = get_model()

    model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss=customized_loss,
                  metrics=['accuracy', dice_coefficient], sample_weight_mode='temporal')
    lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
    csv_logger = CSVLogger('Model/ModelSettings/PigmentLogger.csv')
    model_checkpoint = ModelCheckpoint("Model/ModelSettings/Pigment-weights.hdf5", monitor='val_loss', verbose=1,
                                       save_best_only=True)

    X, y = get_dataset('data/processed/ResizedImages/', 'data/processed/ResizedPigments/')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    h = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test),
                  callbacks=[lr_reducer, csv_logger, model_checkpoint])

    save_reports(h)
    model.load_weights('models/weights.hdf5')
    model.save('models/Pigment-model2.h5')


if __name__ == '__main__':
    train_model()
