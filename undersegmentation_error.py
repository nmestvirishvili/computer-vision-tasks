# -*- coding: utf-8 -*-
"""
Group Number: 18
@author: Janik Hasche, Christian Kehler, Natia Mestvirishvili
"""

from __future__ import annotations
import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from skimage import segmentation


class LabelImage:

    def __init__(self, label_image: npt.NDArray):
        self.label_image = label_image

    def get_segment_indices(self):
        return np.unique(self.label_image.flatten())

    def get_segment_pixels(self, segment_index: int) -> npt.NDArray:
        return np.column_stack(np.where(self.label_image == segment_index))

    def get_segment_area(self, segment_index: int) -> npt.NDArray:
        return self.get_segment_pixels(segment_index).shape[0]

    def get_intersecting_segment_indices(self, pixels: npt.NDArray):
        segment_indices = []
        for pixel in pixels:
            y = pixel[0]
            x = pixel[1]
            segment_indices.append(self.label_image[y, x])
        return np.unique(segment_indices)


def show_image(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


def draw_area_on_image(image, img: npt.NDArray, pixels: npt.NDArray, color: int = 0):
    for pixel in pixels:
        y = pixel[0]
        x = pixel[1]
        img[y, x, color] = min(image[y, x, 0] + 0.3, 1.0)
    return img

def calc_undersegmentation_error(I_TRUTH_SEGMENT, n_segments, compactness):
    image = imageio.imread('0001_rgb.png')

    slic = segmentation.slic(image, n_segments=n_segments, compactness=compactness, start_label=1)
    slic_label_image = LabelImage(slic)

    ground_truth = imageio.imread('0001_label.png')
    ground_truth_label_image = LabelImage(ground_truth)

    segment_indices = ground_truth_label_image.get_segment_indices()
    segment_pixels = ground_truth_label_image.get_segment_pixels(I_TRUTH_SEGMENT)

    boundary_image = segmentation.mark_boundaries(image, slic)
    marked_image = draw_area_on_image(image, boundary_image, segment_pixels)

    intersecting_segment_indices = slic_label_image.get_intersecting_segment_indices(segment_pixels)
    for index in intersecting_segment_indices:
        pixels = slic_label_image.get_segment_pixels(index)
        marked_image = draw_area_on_image(image, marked_image, pixels, 1)

    show_image(marked_image, f'Intersection of segmentation n_segments={n_segments}, compactness={compactness} with ground truth segment={I_TRUTH_SEGMENT}')

    # calc under-segmentation error
    intersecting_segments_area = 0
    for index in intersecting_segment_indices:
        segment_area = slic_label_image.get_segment_area(index)
        intersecting_segments_area += segment_area

    truth_area = ground_truth_label_image.get_segment_area(I_TRUTH_SEGMENT)

    under_segmentation_error = (intersecting_segments_area - truth_area) / truth_area
    return under_segmentation_error

def avg_undersegmentation(n_segments, compactness):
    image = imageio.imread('0001_rgb.png')

    slic = segmentation.slic(image, n_segments=n_segments, compactness=compactness, start_label=1)
    show_image(slic, f'Segmentation n_segments={n_segments}, compactness={compactness}')
    boundary_image = segmentation.mark_boundaries(image, slic)
    show_image(boundary_image, f'Segmentation boundaries on image n_segments={n_segments}, compactness={compactness}')
    
    ground_truth = imageio.imread('0001_label.png')
    ground_truth_label_image = LabelImage(ground_truth)
    segment_indices = ground_truth_label_image.get_segment_indices()
    
    total_undersegmentation_errors = 0
    print(f'Undersegmentation errors for n_segments={n_segments}, compactness={compactness}')
    for segment_val in segment_indices:
        if (segment_val != 0):
            undersegmentation_error = calc_undersegmentation_error(segment_val, n_segments, compactness)
            print(undersegmentation_error)
            total_undersegmentation_errors += undersegmentation_error
    avg_undersegmentation_error = total_undersegmentation_errors/(len(segment_indices)-1)
    print(f'Average undersegmentation error for n_segments={n_segments}, compactness={compactness}: {avg_undersegmentation_error}')   
    return avg_undersegmentation_error

if __name__ == '__main__':
    # Optimal parameters n_segments and compactness for segmentation
    n_segments = 24
    compactness = 50
    avg_undersegmentation(n_segments, compactness)
    
    # See what happens if we increase the the desired number of superpixels n
    avg_undersegmentation(30, compactness)
    avg_undersegmentation(48, compactness)
    
    # Undersegmentation error decreases as we tend to increase the desired number of superpixels n. This behaviour
    # is expected, as undersegmentation error only measures the degree of error in having less segments than needed.
    # However, the fact that undersegmentation error decreases is not neccessirily ideal, because the oversegmentation
    # error might increase if we choose a number of superpixels that is too high.
    
    