# -*- coding: utf-8 -*-
"""
Created on Wed Apr 5 12:09:25 2023

@author: Sourav Saha
"""
import numpy as np

#Temperature distribution
T = np.array([[30, 32, 37, 33], [32, 38, 39, 31],[30, 34, 36, 32],[32, 35, 37, 33]])

stepx = 0.4 #step size along X 
stepy = 0.4 #step size along y
Area = 1.2*1.2 # Area of the plate 

filter = np.array([[stepx*stepy/4, stepx*stepy/4],[stepx*stepy/4, stepx*stepy/4]]) #filter 
trap_filter = np.flip(filter)

# Convolution function 
def convolution2d(image, kernel):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return new_image

P = convolution2d(T, trap_filter)

Integral = np.sum(P)

print('The integration value = ', Integral,'Cm^2')
print("Average temperature: ",Integral/Area,"C")


