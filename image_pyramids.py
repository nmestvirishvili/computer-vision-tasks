"""
@authors: Christian Kehler, Janik Hasche, Natia Mestvirishvili
"""

import imageio
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import skimage.transform.pyramids as pyramids
from skimage import transform

def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


image = imageio.imread('visual_attention.png')
image = color.rgb2gray(color.rgba2rgb(image))

center_pyramid = tuple(pyramids.pyramid_gaussian(image, max_layer=4, sigma=9))
surround_pyramid = tuple(pyramids.pyramid_gaussian(image, max_layer=4, sigma=16))

#Display Center pyramid
show_image(center_pyramid[1], "Center pyramid: Level 1")
show_image(center_pyramid[2], "Center pyramid: Level 2")
show_image(center_pyramid[3], "Center pyramid: Level 3")
show_image(center_pyramid[4], "Center pyramid: Level 4")

#Display Surround pyramid
show_image(surround_pyramid[1], "Surround pyramid: Level 1")
show_image(surround_pyramid[2], "Surround pyramid: Level 2")
show_image(surround_pyramid[3], "Surround pyramid: Level 3")
show_image(surround_pyramid[4], "Surround pyramid: Level 4")

#Calculate on-off and off-on pyramids
onOffPyramid = []
offOnPyramid = []
for index in range(0, 5):
    onOff = center_pyramid[index] - surround_pyramid[index]
    onOff = np.clip(onOff, 0, 1)
    offOn = surround_pyramid[index] - center_pyramid[index]
    offOn = np.clip(offOn, 0, 1)
    onOffPyramid.append(onOff)
    offOnPyramid.append(offOn)

#Display on-off and off-on pyramids
show_image(onOffPyramid[1], "On-off pyramid: Level 1")
show_image(onOffPyramid[2], "On-off pyramid: Level 2")
show_image(onOffPyramid[3], "On-off pyramid: Level 3")
show_image(onOffPyramid[4], "On-off pyramid: Level 4")

show_image(offOnPyramid[1], "Off-on pyramid: Level 1")
show_image(offOnPyramid[2], "Off-on pyramid: Level 2")
show_image(offOnPyramid[3], "Off-on pyramid: Level 3")
show_image(offOnPyramid[4], "Off-on pyramid: Level 4")

#resize pyramid levels
map_width = onOffPyramid[0].shape[1]
map_height = onOffPyramid[0].shape[0]

for index in range(1, 5):
     onOffPyramid[index] = transform.resize(onOffPyramid[index], (map_height, map_width))
     offOnPyramid[index] = transform.resize(offOnPyramid[index], (map_height, map_width))

#create feature and conspicuity maps
feature_map_onOff = np.zeros([map_height, map_width])
feature_map_offOn = np.zeros([map_height, map_width])
conspicuity_map = np.zeros([map_height, map_width])
for index_width in range(0, map_width):
    for index_height in range(0, map_height):
        #create on-off feature map
        pixelValue_onOff = onOffPyramid[0][index_height][index_width]
        pixelValue_onOff += onOffPyramid[1][index_height][index_width]
        pixelValue_onOff += onOffPyramid[2][index_height][index_width]
        pixelValue_onOff += onOffPyramid[3][index_height][index_width]
        pixelValue_onOff += onOffPyramid[4][index_height][index_width]
        pixelValue_onOff = pixelValue_onOff/5
        feature_map_onOff[index_height][index_width] = pixelValue_onOff
        #create off-on feature map
        pixelValue_offOn = offOnPyramid[0][index_height][index_width]
        pixelValue_offOn += offOnPyramid[1][index_height][index_width]
        pixelValue_offOn += offOnPyramid[2][index_height][index_width]
        pixelValue_offOn += offOnPyramid[3][index_height][index_width]
        pixelValue_offOn += offOnPyramid[4][index_height][index_width]
        pixelValue_offOn = pixelValue_offOn/5
        feature_map_offOn[index_height][index_width] = pixelValue_offOn
        conspicuity_map[index_height][index_width] = pixelValue_offOn + pixelValue_onOff / 2

show_image(feature_map_onOff, "On-off feature map")
show_image(feature_map_offOn, "Off-on feature map")
show_image(conspicuity_map, "Conspicuity map")     

# Gaussian pyramid gives us the opportunity to use gaussian filters, which help us smooth the image and 
# get rid of the noise, which could skew the results of saliency map. Furthermore, gaussian filters 
# give us the opportunity to easily adjust the parameters according to our needs.
# For example, changing the center and surroinding window sizes would require changing
# the algorithm in some way, while the same adjustment to the saliency map using gaussian pyramids could be made by 
# just changing the deviations of gaussian distributions. Furthermore, when substracting the two gaussians
# also highlights the edges that are present in the image.