from PIL import Image
import matplotlib.pyplot as plt
import os
import re

SIZE = (256, 256)


def resize_image(src: str, dest: str = 'C:/Users/user/PycharmProjects/VKR/data/TestImages/output.png'):
    image = Image.open(src)
    image = image.crop((0, 200, 640, 840))
    image = image.resize(SIZE)
    plt.imshow(image, cmap='gray')
    image.save(dest)


if __name__ == '__main__':
    path_name = 'C:/Users/user/PycharmProjects/VKR/data/labeled_images/6/Pigment/'
    files = [path_name + n for n in os.listdir(path_name)]
    files = sorted(files,
                   key=lambda file: int(file.split('/')[-1].split('.')[0]))
    print(files)
    for i, f in enumerate(files):
        resize_image(f, 'C:/Users/user/PycharmProjects/VKR/data/labeled_images/6/Resize/' + str(i+1) + '.png')
