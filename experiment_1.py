import numpy as np
import matplotlib.pyplot as plt
import math

outfile_path = "Experiment_1"
#Part 1.a
f = [2, 3, 4, 4] 

f_transformed = np.fft.fft(f, norm='forward')
f_reverse_transform = np.fft.ifft(f_transformed, norm='forward')

# print(f_transformed)
# print(f_reverse_transform)

#Part 1.b
cos_sample_size = 128
cycles = 8 #number of times function will complete 1 cycle. Cycle is repeated pattern
u = 8
period = (2*math.pi)/(2*math.pi*u) # = 1/u, period is x distance required to complete a cycle

#get max x value 
max_x = period*cycles

#calculate sample points within interval
sample_points = np.linspace(0, max_x, 128)

#sample cos function
cos_samples = []
for x in sample_points:
    cos_samples.append((math.cos(2*math.pi*u*x))*(-1**x))#multiply by (-1^x) to center fourier transform

cos_samples = np.array(cos_samples)

#plot figure
plt.figure(figsize=(10, 3))
plt.plot(sample_points, cos_samples, 'o')
plt.title("Sampled Cosine")
plt.savefig(f"{outfile_path}\Sampled_Cosine.png")

#compute DFT of cos function
cos_dft = np.fft.fft(cos_samples, norm='forward')

#plot real
plt.figure(figsize=(10, 3))
plt.plot(sample_points, cos_dft.real)
plt.title("Cosine DFT Real")
plt.savefig(f"{outfile_path}\Cosine_DFT_Real.png")

#plot imaginary
plt.figure(figsize=(10, 3))
plt.plot(sample_points, cos_dft.imag)
plt.title("Cosine DFT Imaginary")
plt.savefig(f"{outfile_path}\Cosine_DFT_Imaginary.png")

#calculate magnitude and phase of cos dft
cos_mag = []
cos_phase = []

for real, imag in zip(cos_dft.real, cos_dft.imag):
    mag = math.sqrt(real**2+imag**2)
    phase = math.atan(imag/real)

    cos_mag.append(mag)
    cos_phase.append(phase)

#plot mag
plt.figure(figsize=(10, 3))
plt.plot(sample_points, cos_mag)
plt.title("Magnitude of Cosine DFT")
plt.savefig(f"{outfile_path}\Cosine_DFT_Mag.png")

#plot phase
plt.figure(figsize=(10, 3))
plt.plot(sample_points, cos_phase)
plt.title("Phase of Cosine DFT")
plt.savefig(f"{outfile_path}\Cosine_DFT_Phase.png")


#Part 1.c
with open("Rect_128.txt") as file:
    rect_values = file.readlines()
    rect_values = [float(value.strip()) for value in rect_values]
    rect_values = [(-1**value)*value for value in rect_values]#center fourier transform

#calculate x-values
#rectangle centered on x = 0
rect_x_values = list(range(-len(rect_values)//2, len(rect_values)//2))

#plot rectangle
plt.figure(figsize=(10, 3))
plt.plot(rect_x_values, rect_values, 'o')
plt.title("Rectangle")
plt.savefig(f"{outfile_path}\Rectangle.png")

#compute dft of rectangle
rect_dft = np.fft.fft(rect_values, norm='forward')

#plot rect dft real
plt.figure(figsize=(10, 3))
plt.plot(rect_x_values, rect_dft.real)
plt.title("Rectangle DFT Real")
plt.savefig(f"{outfile_path}\Rectangle_DFT_Real.png")

#plot rect dft imag
plt.figure(figsize=(10, 3))
plt.plot(rect_x_values, rect_dft.imag)
plt.title("Rectangle DFT Imaginary")
plt.savefig(f"{outfile_path}\Rectangle_DFT_Imag.png")

#calculate magnitude and phase of rectangle dft
rect_mag = []
rect_phase = []

for real, imag in zip(rect_dft.real, rect_dft.imag):
    mag = math.sqrt(real**2+imag**2)
    phase = math.atan(imag/real)

    rect_mag.append(mag)
    rect_phase.append(phase)

#plot mag
plt.figure(figsize=(10, 3))
plt.plot(rect_x_values, rect_mag)
plt.title("Magnitude of Rectangle DFT")
plt.savefig(f"{outfile_path}\Rectangle_DFT_Mag.png")

#plot phase
plt.figure(figsize=(10, 3))
plt.plot(rect_x_values, rect_phase)
plt.title("Phase of Rectangle DFT")
plt.savefig(f"{outfile_path}\Rectangle_DFT_Phase.png")