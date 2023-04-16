from PIL import Image
import matplotlib.pyplot as plt
import os

SIZE = (256, 256)


def resize_image(src: str, dest: str):
    image = Image.open(src)
    image = image.crop((0, 200, 640, 840))
    image = image.resize(SIZE)
    plt.imshow(image, cmap='gray')
    image.save(dest)


def preprocessing_images(src: str, dest: str):
    files = [src + n for n in os.listdir(src)]
    files = sorted(files, key=lambda file: int(file.split('/')[-1].split('.')[0]))
    for i, f in enumerate(files):
        resize_image(f, dest + str(i + 1) + '.png')


if __name__ == '__main__':
    preprocessing_images('../../data/labeled_images/6/Pigment/',
                         '../../data/labeled_images/6/Resize/')
