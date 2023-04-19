# -*- coding: utf-8 -*-
"""
Created on Wed Apr 5 12:09:25 2023

@author: Sourav Saha
Adapted by Klemens Iten
"""
import numpy as np

#Temperature distribution
T = np.array([[30, 32, 37, 33], [32, 38, 39, 31],[30, 34, 36, 32],[32, 35, 37, 33]])

stepx = 0.4 #step size along X 
stepy = 0.4 #step size along y
Area = 1.2*1.2 # Area of the plate 
Area_simpson = 0.8*0.8*4 # Area of the simpson integration domain: Compensates for Area counted twice under 1-stride Simpson

trap_filter = np.array([[stepx*stepy/4, stepx*stepy/4],[stepx*stepy/4, stepx*stepy/4]]) #filter 
trap_filter = np.flip(trap_filter)

simp_filter = np.array([[1, 4, 1],[4, 16, 4], [1, 4, 1]]) * stepx*stepy/9 #filter 
simp_filter = np.flip(simp_filter)

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

P_trap = convolution2d(T, trap_filter)
P_simp_trap_1 = convolution2d(T[0:-1,0:-1], simp_filter)    # Split solar panel into 3 different regions
P_simp_trap_2 = convolution2d(T[:,-2:], trap_filter)        # for combined Simpson-trapezoidal
P_simp_trap_3 = convolution2d(T[-2:,:-1], trap_filter)
P_simp = convolution2d(T, simp_filter)

Integral_trap = np.sum(P_trap)
Integral_simp_trap = np.sum(P_simp_trap_1) + np.sum(P_simp_trap_2) + np.sum(P_simp_trap_3)
Integral_simp = np.sum(P_simp)

print('The integration value with trapezoidal = ', Integral_trap,'Cm^2')
print("Average temperature with trapezoidal: ",Integral_trap/Area,"C")

print('The integration value with Simpson = ', Integral_simp_trap,'Cm^2')
print("Average temperature with Composite Simpson-Trapezoidal: ",Integral_simp_trap/Area,"C")

print('The integration value with Simpson = ', Integral_simp,'Cm^2')
print("Average temperature with Simpson: ",Integral_simp/Area_simpson,"C")
