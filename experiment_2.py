import numpy as np
from helpers import fft2D, generate_img, magnitude_2D
import cv2
import os

outfile_path = 'Experiment_2'
# testing for 2D fft
test_array = np.array([[2, 3, 4, 4], 
                      [2, 3, 4, 4], 
                      [2, 3, 4, 4], 
                      [2, 3, 4, 4]])
# for value in test_array:
#     print(value)
# test_array_fft = fft2D(0, 0, test_array, -1)

# test_array_mag = magnitude_2D(test_array_fft)
# print(test_array_fft)
# print(test_array_mag)
# test_array_ifft = fft2D(0, 0, test_array_fft, 1)

# print(test_array_fft)
# print('\n')
# print(test_array_ifft)

#generate images for parts 2.a-2.c
img_2a = generate_img(512, 32)
img_2b = generate_img(512, 64)
img_2c = generate_img(512, 128)

cv2.imwrite(os.path.join(outfile_path, '2a.jpg'), img_2a)
cv2.imwrite(os.path.join(outfile_path, '2b.jpg'), img_2b)
cv2.imwrite(os.path.join(outfile_path, '2c.jpg'), img_2c)

