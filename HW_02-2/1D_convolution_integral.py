# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:18:17 2023

@author: Sourav Saha
"""

import numpy as np
from scipy import integrate
#generate the data for integration

u = 2000 #m/s (downwards velocity)
m_o = 170000 #kg
q = 2500 #kg/s
g = 9.8 #m/s^2

step = (4-0)/8

t_low = 0 #lower integration limit
t_up = 4 #upper integration limit


t = np.arange(t_low,(t_up+step),step)

v = -(u*np.log(m_o/(m_o-g*t))-g*t)

trap_filter = np.array([step/2, step/2]) #trapezoidal

###### Trapezoidal Method #########
trap_filter = np.flip(trap_filter)
y2 = np.sum(np.convolve(trap_filter,v,mode='valid'))


print("Distance covered:", y2)

gap = t_up-t_low
true_trap = np.trapz(v,t,gap)

print("Trapezoidal Method Distacne: ",true_trap)

sim_filter = np.array([step/3, 4*step/3, step/3]) #Simpson's Rule


#@njit
# Define the convolution function
def Simpson_convolution(x, filter):
    # Get the length of the input signal and the filter
    m, n = len(x), len(filter)
    m = int(m/2)
    y = np.zeros(m)
    # Perform the convolution
    stride = 0
    for i in range(m):
        y[i] = np.sum(filter * x[i+stride:i+n+stride])
        stride = stride + 1    
    print(y)
    S = np.sum(y)
    return S

y = Simpson_convolution(v, sim_filter)
#print("Velocity:", v)
print("Covered Distance:", y)
P = integrate.simpson(v, t)
print("Simpson's method distance: ",P)