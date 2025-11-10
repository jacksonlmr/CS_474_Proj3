import numpy as np
import math
from helpers import magnitude, phase, center_spectrum, create_figure

outfile_path = "Experiment_1"
#Part 1.a
f = np.array([2, 3, 4, 4]) 
f_x_values = np.array([0, 1, 2, 3])

# f = center_spectrum(f)
# print(f"f centered: {f}")

f_transformed = np.fft.fft(f, norm='forward')
f_reverse_transform = np.fft.ifft(f_transformed, norm='forward')
print(f"f transformed: {f_transformed}\ninverse transform: {f_reverse_transform}")

#compute magnitude
f_mag = magnitude(f_transformed)

#plot function, real, imaginary, and magnitude
create_figure('f', f_x_values, f, outfile_path, True)
create_figure('f_real', f_x_values, f_transformed.real, outfile_path, True)
create_figure('f_imag', f_x_values, f_transformed.imag, outfile_path, True)
create_figure('f_mag', f_x_values, f_mag, outfile_path, True)

#Part 1.b
cos_sample_size = 128
cycles = 8 #number of times function will complete 1 cycle. Cycle is repeated pattern
u = 8
period = (2*math.pi)/(2*math.pi*u) # = 1/u, period is x distance required to complete a cycle

#get max x value 
max_x = period*cycles

#calculate sample points within interval
sample_points = np.linspace(0, max_x, 128, endpoint=False)

#sample cos function
cos_samples = []
for x in sample_points:
    cos_samples.append((math.cos(2*math.pi*u*x)))

cos_samples = np.array(cos_samples)

#plot cosine samples
create_figure('Sampled_Cosine', sample_points, cos_samples, outfile_path, True)

#compute DFT of cos function using f[x](-1)^x
cos_samples = center_spectrum(cos_samples)

cos_dft = np.fft.fft(cos_samples, norm='forward')
cos_dft_freq = np.fft.fftfreq(cos_dft.size)

#calculate magnitude and phase of cos dft
cos_mag = magnitude(cos_dft)
cos_phase = phase(cos_dft)

#plot real
create_figure('Cosine_DFT_Real', cos_dft_freq, cos_dft.real, outfile_path, True)

#plot imaginary 
create_figure('Cosine_DFT_Imaginary', cos_dft_freq, cos_dft.imag, outfile_path, True)

#plot mag
create_figure('Cosine_DFT_Mag', cos_dft_freq, cos_mag, outfile_path, True)

#plot phase
create_figure('Cosine_DFT_Phase', cos_dft_freq, cos_phase, outfile_path, False)



#Part 1.c
with open("Rect_128.txt") as file:
    rect_values = file.readlines()
    rect_values = [float(value.strip()) for value in rect_values]
    rect_values = np.array(rect_values)

#calculate x-values
#rectangle centered on x = 0
rect_x_values = np.array(list(range(-len(rect_values)//2, len(rect_values)//2)))

#plot rectangle
create_figure('Rectangle', rect_x_values, rect_values, outfile_path, True)

#compute dft of rectangle
rect_values = center_spectrum(rect_values)
rect_dft = np.fft.fft(rect_values, norm='forward')
# rect_dft_freq = np.fft.fftfreq(rect_dft)

#calculate magnitude and phase of rectangle dft
rect_mag = magnitude(rect_dft)
rect_phase = phase(rect_dft)

#plot rect dft real
create_figure('Rectangle_DFT_Real', rect_x_values, rect_dft.real, outfile_path, False)

#plot rect dft imag
create_figure('Rectangle_DFT_Imag', rect_x_values, rect_dft.imag, outfile_path, False)

#plot mag
create_figure('Rectangle_DFT_Mag', rect_x_values, rect_mag, outfile_path, False)

#plot phase
create_figure('Rectangle_DFT_Phase', rect_x_values, rect_phase, outfile_path, False)