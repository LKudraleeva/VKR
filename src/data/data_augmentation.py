import os
import albumentations as A
from skimage import io
from PIL import Image


def get_images(src_im, src_label):
    images_filenames_train = [src_im + p for p in os.listdir(path=src_im)]
    labels_filenames_train = [src_label + p for p in os.listdir(path=src_label)]

    labels_filenames_train = sorted(labels_filenames_train, key=lambda file: int((file.split('/')[-1]).split('.')[0]))
    images_filenames_train = sorted(images_filenames_train, key=lambda file: int((file.split('/')[-1]).split('.')[0]))

    return [io.imread(image) for image in images_filenames_train], [io.imread(lab) for lab in labels_filenames_train]


def augmentation(pictures, masks):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([A.ElasticTransform(),
                 A.OpticalDistortion(),
                 A.GridDistortion()], p=1),
        A.OneOf([A.RandomBrightnessContrast(),
                 A.RandomGamma()], p=1),

    ])
    for idx, (im, l) in enumerate(zip(pictures, masks)):
        augmented = transform(image=im, mask=l)
        im = Image.fromarray(augmented['image'])
        label = Image.fromarray(augmented['mask'])
        im.save('../../data/dataset/AugmentatedImages/' + str(idx + 1) + '.png')
        label.save('../../data/dataset/AugmentatedPigment/' + str(idx + 1) + '.png')


if __name__ == '__main__':
    images, labels = get_images('../../data/dataset/ResizedImages/', '../../data/dataset/ResizedPigment/')
    augmentation(images, labels)
