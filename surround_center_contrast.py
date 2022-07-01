# Group members: Christian Kehler, Janik Hasche, Natia Mestvirishvili

import imageio
import matplotlib.pyplot as plt
import skimage.transform
from skimage import color
import numpy as np

def surroundCenterContrasts(center_window_size, surround_window_size):
    image = imageio.imread('visual_attention_ds.png')
    image = color.rgba2rgb(image)
    image = color.rgb2gray(image)
    
    integral_image = skimage.transform.integral_image(image)
    
    y_size = image.shape[0]
    x_size = image.shape[1]
    
    center_average_image = np.zeros((y_size, x_size))
    surround_average_image = np.zeros((y_size, x_size))
    
    frame_to_cut_x = 0
    frame_to_cut_y = 0
    for x in range(x_size):
        for y in range(y_size):
            x_start = int(x - ((surround_window_size - 1) / 2))
            x_end = x_start + surround_window_size
            y_start = int(y - ((surround_window_size - 1) / 2))
            y_end = y_start + surround_window_size
            if (x_start < 0 or x_end >= x_size):
                frame_to_cut_x += 1
                continue
            if y_start < 0 or y_end >= y_size:
                frame_to_cut_y += 1
                continue
    
            n = surround_window_size ^ 2
            average = skimage.transform.integrate(integral_image, (y_start, x_start), (y_end, x_end)) / n
            surround_average_image[y][x] = average
    
    for x in range(x_size):
        for y in range(y_size):
            x_start = int(x - ((center_window_size - 1) / 2))
            x_end = x_start + center_window_size
            y_start = int(y - ((center_window_size - 1) / 2))
            y_end = y_start + center_window_size
            if x_start < 0 or x_end >= x_size:
                continue
            if y_start < 0 or y_end >= y_size:
                continue
    
            n = center_window_size ^ 2
            average = skimage.transform.integrate(integral_image, (y_start, x_start), (y_end, x_end)) / n
            center_average_image[y][x] = average
    
    # crop images to remove empty pixels  
    surround_image_edge = int((surround_window_size - 1) / 2)
    crop_start_x = surround_image_edge
    crop_end_x = x_size - surround_image_edge - 1
    crop_start_y = surround_image_edge
    crop_end_y = y_size - surround_image_edge - 1
    surround_average_image = surround_average_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
    center_average_image = center_average_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
    
    surround_minus_center = surround_average_image - center_average_image
    
    plt.imshow(surround_average_image, cmap='gray', interpolation='nearest')
    plt.show()
    plt.imshow(center_average_image, cmap='gray', interpolation='nearest')
    plt.show()
    plt.imshow(surround_minus_center, cmap='gray', interpolation='nearest')
    plt.show()

center_window_size = 11
surround_window_size = 21
surroundCenterContrasts(center_window_size, surround_window_size)
surroundCenterContrasts(3, 7)
surroundCenterContrasts(31, 51)

# As per the observation on the sizing of the center and surrounding windows, the larger the
# sizes of the windows are, the blurrier the image egets. 