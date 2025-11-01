import numpy as np
import matplotlib 

f = [2, 3, 4, 4] 

f_transformed = np.fft.fft(f, norm='forward')
f_reverse_transform = np.fft.ifft(f_transformed, norm='forward')
print(f_transformed)
print(f_reverse_transform)