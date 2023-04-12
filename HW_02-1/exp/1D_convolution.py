# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:18:17 2023

@author: Sourav Saha
"""

import numpy as np
from numba import njit
# Define the input signal and the filter
x = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45])
filter = np.array([-0.5, 0, 0.5])

@njit
# Define the convolution function
def convolution(x, filter):
    # Get the length of the input signal and the filter
    m, n = len(x), len(filter)
    m = m - 2
    # Initialize the output signal
    y = np.zeros(m)

    # Perform the convolution
    for i in range(m):
        y[i] = np.sum(filter * x[i:i+n])
    return y

# Test the convolution function
y = convolution(x, filter)
print("Input signal:", x)
print("Filter:", filter)
print("Intermediate derivative:", y)

# # Modify the filter at the boundary

y_l = np.zeros(1)
new_filter_l = np.array([-1., 1., 0])
y_l[0] = np.sum(new_filter_l * x[0:3])
print("left boundary:", y_l)

y_r = np.zeros(1)
new_filter_r = np.array([-1., 1., 0])
y_r[0] = np.sum(new_filter_r * x[6:9])
print("right boundary:", y_r)

Y = np.concatenate([y_l, y, y_r],axis =0)

print("Total derivative:", Y)

###### Alternate way only works in 1D #########
filter = np.flip(filter)
y2 = np.convolve(filter,x,mode='valid')
print("Alternate intermediate derivative:", y2)
Y2 = np.concatenate([y_l, y2, y_r],axis =0)
print("Alternate Total derivative:", y2)