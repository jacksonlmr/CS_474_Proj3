import numpy as np
from helpers import fft2D

test_array = np.array([[2, 3, 4, 4], 
                      [2, 3, 4, 4], 
                      [2, 3, 4, 4], 
                      [2, 3, 4, 4]])

fft2D(0, 0, test_array, 1)

print(test_array)
# #loop through rows
# for row in test_array:
#     print(row)

# #loop through columns
# for x in range(test_array.shape[1]):
#     col = test_array[:, x]
#     print(col)