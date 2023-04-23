import numpy as np
import matplotlib.pyplot as plt

def RationalQuadratic(x, alpha, k):
  x_n = x # / dx normalize x 
  kernel = (1 + (x_n ** 2) / (2 * alpha * (k ** 2))) ** (-alpha)
  return kernel

x_sample = np.array([0.3, 0.5, 0.58, 0.64, 0.7, 0.8, 0.9, 1.0])
y_sample = np.array([630, 1098, 1641, 1733, 2000, 2885, 3508, 4140])
x = np.linspace(x_sample.min(), x_sample.max(), num=1000)
y = np.zeros_like(x)

x_sample_norm = (x_sample - x_sample.mean()) / x_sample.std()
print(x_sample_norm)
x_norm = (x - x_sample.mean()) / x_sample.std()

alpha = 2
k = 0.1

for idx, y_i in enumerate(y):
    kernel_sum = 0
    for x_sample_i, y_sample_i in zip(x_sample_norm, y_sample):
       y_i += y_sample_i*RationalQuadratic(x_norm[idx]-x_sample_i, alpha, k)
       kernel_sum += RationalQuadratic(x_norm[idx]-x_sample_i, alpha, k)
    y[idx] = y_i / kernel_sum

plt.plot(x, y)
plt.plot(x_sample, y_sample, 'o')
plt.show()
