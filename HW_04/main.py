from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy import signal

from fft_433 import fft as fft2
from fft_433 import plot_fft, maf, plot_maf

import numpy as np
import matplotlib.pyplot as plt

def sinc(X: float) -> float:
    return 1.0 if X==0 else np.sin(X)/X

def lowpass_filter(length, sampling_rate, cutoff_freq = 65) -> np.ndarray:
    """https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter"""
    cutoff_freq = cutoff_freq / sampling_rate
    filter = np.zeros((length,),dtype=float)
    for idx in range(length):
        filter[idx] = 2*cutoff_freq*sinc(2*cutoff_freq*(length/2-idx))
    return filter

sampling_rate, data = wavfile.read("guitar_drum.wav")
data_l = data[:, 0]
data_r = data[:, 1]
time = np.arange(0, len(data_l) / sampling_rate, 1 / sampling_rate)
frq_l, Y_l = fft2(time, data_l)
frq_r, Y_r = fft2(time, data_r)
plot_fft(time, data_l, frq_l, Y_l)
plot_fft(time, data_r, frq_r, Y_r)

filter_length = 100001

lowpass = np.empty_like(data_l)
filter = lowpass_filter(filter_length, sampling_rate, cutoff_freq=400)

lowpass = signal.fftconvolve(data_l, filter, mode='same')
lowpass /= np.mean(np.abs(lowpass)) / np.mean(np.abs(data))

frq_low, Y_low = fft2(time, lowpass)
plot_fft(time, lowpass, frq_low, Y_low)

filter = - filter # spectral inversion
filter[int(np.ceil(filter_length/2))] += 1

highpass = signal.fftconvolve(data_l, filter, mode='same')
highpass /= np.mean(np.abs(highpass)) / np.mean(np.abs(data))

frq_high, Y_high = fft2(time, highpass)
plot_fft(time, highpass, frq_high, Y_high)

plt.figure()
plt.grid()
plt.plot(time, lowpass, 'b', label='Low Pass')
plt.xlim(0, 1)
plt.legend(loc='upper center')
plt.show()

wavfile.write("guitar_drum_unchanged.wav", sampling_rate, data.astype(np.int16))
wavfile.write("guitar_drum_lowpass.wav", sampling_rate, lowpass.astype(np.int16))
wavfile.write("guitar_drum_highpass.wav", sampling_rate, highpass.astype(np.int16))

print(1)