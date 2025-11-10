import numpy as np
from helpers import fft2D, generate_img, magnitude_2D, scale_magnitude, mapValues
import cv2
import os

outfile_path = 'Experiment_2'
# testing for 2D fft
# test_array = np.array([[2, 3, 4, 4], 
#                       [2, 3, 4, 4], 
#                       [2, 3, 4, 4], 
#                       [2, 3, 4, 4]])
test_array = generate_img(8, 4)
print(test_array)
print()
test_array_fft = fft2D(test_array, -1)
print(test_array_fft)
print()
test_array_fft_mag = magnitude_2D(test_array_fft)
print(test_array_fft_mag)
print()
test_array_fft_mag_scaled = scale_magnitude(test_array_fft_mag)
print(test_array_fft_mag_scaled)
print()
test_array_fft_mag_scaled = mapValues(test_array_fft_mag_scaled)
print(test_array_fft_mag_scaled)

cv2.imwrite(os.path.join(outfile_path, 'test_array.jpg'), test_array)
cv2.imwrite(os.path.join(outfile_path, 'test_array_mag.jpg'), test_array_fft_mag_scaled)
#generate images for parts 2.a-2.c
img_2a = generate_img(512, 32)
img_2b = generate_img(512, 64)
img_2c = generate_img(512, 128)

cv2.imwrite(os.path.join(outfile_path, '2a.jpg'), img_2a)
cv2.imwrite(os.path.join(outfile_path, '2b.jpg'), img_2b)
cv2.imwrite(os.path.join(outfile_path, '2c.jpg'), img_2c)

#calculate ffts
img_2a_fft = fft2D(img_2a, -1)
img_2b_fft = fft2D(img_2b, -1)
img_2c_fft = fft2D(img_2c, -1)

#calculate magnitudes
img_2a_fft_mag = magnitude_2D(img_2a_fft)
img_2b_fft_mag = magnitude_2D(img_2b_fft)
img_2c_fft_mag = magnitude_2D(img_2c_fft)

#scale magnitudes
img_2a_fft_mag_scaled = mapValues(scale_magnitude(img_2a_fft_mag))
img_2b_fft_mag_scaled = mapValues(scale_magnitude(img_2b_fft_mag))
img_2c_fft_mag_scaled = mapValues(scale_magnitude(img_2c_fft_mag))

#visualize magnitudes unshifted
cv2.imwrite(os.path.join(outfile_path, '2a_mag_unshifted.jpg'), img_2a_fft_mag_scaled)
cv2.imwrite(os.path.join(outfile_path, '2b_mag_unshifted.jpg'), img_2b_fft_mag_scaled)
cv2.imwrite(os.path.join(outfile_path, '2c_mag_unshifted.jpg'), img_2c_fft_mag_scaled)

#calculate shifted magnitudes
