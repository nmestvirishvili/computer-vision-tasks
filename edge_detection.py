"""
@authors: Christian Kehler, Janik Hasche, Natia Mestvirishvili
"""

import imageio
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import skimage.filters
import numpy as np

def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def draw_hist(img, title):
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 1))
    plt.figure()
    plt.title(title)
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0.0, 1.0])
    plt.plot(bin_edges[0:-1], histogram)  
    plt.show()
    
    
image = imageio.imread('woman.png')
show_image(image, 'image')

image_noise = skimage.util.random_noise(image, mode='gaussian', var=0.01)
show_image(image_noise, 'Noisy woman')

image_noise_smoothed = skimage.filters.gaussian(image_noise, sigma=1)
show_image(image_noise_smoothed, 'Gaussian filtered woman')

image_sobel_noisy = skimage.filters.sobel(image_noise)
show_image(image_sobel_noisy, 'Noisy woman Sobel')

image_sobel_smoothed = skimage.filters.sobel(image_noise_smoothed)
show_image(image_sobel_smoothed, 'Gaussian filtered Sobel')

draw_hist(image_sobel_noisy, 'Noisy Sobel Histogram')
draw_hist(image_sobel_smoothed, 'Noisy Sobel Histogram')

binary = np.zeros_like(image_sobel_noisy)
binary[image_sobel_noisy > 0.15] = 1.0
show_image(binary, 'mask noisy')

binary = np.zeros_like(image_sobel_smoothed)
binary[image_sobel_smoothed > 0.065] = 1.0
show_image(binary, 'mask smoothed')

# When there is a substantial noise in the image and we use an edge-detecting filter on it, 
# the filter picks up almost every large change in intensity, including the noise. However, if a smoothing
# filter is first applied to the image, the averaging of the pixels in regions including noise gets rid of the
# outliers (which are typically a small number of pixels per region with either very high or very low intensities compared to neighbourhood).
# Subsequently, the edge detection filter no longer picks up on the noise as much as before. 
