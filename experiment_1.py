import numpy as np
import matplotlib.pyplot as plt
import math

#Part 1.a
f = [2, 3, 4, 4] 

f_transformed = np.fft.fft(f, norm='forward')
f_reverse_transform = np.fft.ifft(f_transformed, norm='forward')

print(f_transformed)
print(f_reverse_transform)

#Part 1.b
cos_sample_size = 128
cycles = 8
u = 8
period = 2*math.pi*u

#get max x value 
max_x = (2*math.pi*cycles)/period

#calculate sample points within interval
sample_points = np.linspace(0, max_x, 128)

#sample cos function
cos_samples = []
for x in sample_points:
    cos_samples.append(math.cos(2*math.pi*u*x))

cos_samples = np.array(cos_samples)

plt.plot(sample_points, cos_samples, 'o')
plt.show()