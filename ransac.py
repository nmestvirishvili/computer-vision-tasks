'''
Group Number: 18
Team Members:
1) Janik Hasche
2) Christian Kehler
3) Natia Mestvirishvili
'''
#Task1

import imageio
import matplotlib.pyplot as plt
from skimage import color
from skimage.feature import canny
from skimage.transform import hough_circle
from skimage.transform import hough_circle_peaks
import matplotlib.patches as patches
import numpy as np


def show_image(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


# Read coins image and convert to grayscale
image = imageio.imread('coins.jpg')
image = color.rgb2gray(image)
show_image(image, 'original image converted to grayscale')

# data from coin diameter table
coin_names = [
    '10 marks',
    '5 marks',
    '1 mark',
    '50 penni',
    '10 penni'
]
coin_diameters_mm = np.array([
    27.25,
    24.50,
    22.25,
    19.70,
    16.30,
])

# calculate radius of each coin and convert from mm to pixel
coin_radii_mm = coin_diameters_mm / 2
coin_radii_px = coin_radii_mm / 0.12

# apply canny edge detector
canny_image = canny(image)
show_image(canny_image, 'canny image')

# calculate hough transform and  how result for each radius
hough_images = hough_circle(canny_image, coin_radii_px)
for i in range(len(hough_images)):
    show_image(hough_images[i], 'hough image for ' + coin_names[i])

# superimpose detected circles on the original image:
# go through all 5 generated hough images, get the two peaks for each and apply circle patch to plot
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
for i in range(len(hough_images)):
    accums, cx, cy, radii = hough_circle_peaks([hough_images[i]], [coin_radii_px[i]], num_peaks=2, normalize=True)
    circle = patches.Circle((cx[0], cy[0]), radii[0], alpha=0.1, fc='red')
    ax.add_patch(circle)
    circle = patches.Circle((cx[1], cy[1]), radii[1], alpha=0.1, fc='red')
    ax.add_patch(circle)
plt.title('Detected coins with red patches')
plt.show()
