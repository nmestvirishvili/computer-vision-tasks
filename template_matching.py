"""
@authors: Christian Kehler, Janik Hasche, Natia Mestvirishvili
"""

import imageio
import matplotlib.pyplot as plt
from skimage import color
from skimage.feature import match_template
import numpy as np


def read_to_grayscale(image_path: str):
    image = imageio.imread(image_path)
    image = color.rgb2gray(image)
    return image


def main():
    template = read_to_grayscale('coco264316clock.jpg')
    image = read_to_grayscale('coco264316.jpg')
    matching = match_template(image, template)

    plt.imshow(template, cmap='gray')
    plt.show()

    plt.imshow(image, cmap='gray')
    plt.show()

    plt.imshow(matching, cmap='gray')
    plt.show()

    template_flipped = np.fliplr(template)
    matching_flipped = match_template(image, template_flipped)

    plt.imshow(template_flipped, cmap='gray')
    plt.show()

    plt.imshow(image, cmap='gray')
    plt.show()

    plt.imshow(matching_flipped, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()



