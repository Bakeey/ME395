#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
"""
Some helpful basic signal processing functions, 
made by Klemens Iten for ME433 Advanced Mechatronics at Northwestern
based on templates by Nick Marchuk
"""

def plot_fft(t,y,frq,Y,title = None):
    """
    Plots signal over time as well as frequency spectrum over frequency range.
    Inputs:
        t (time as list), 
        data (signal data as list)
        frq (one side frequency range of signal), 
        Y (fft of signal)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    ax1.plot(t,y,'b')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax2.loglog(frq,abs(Y),'b') # plotting the fft
    ax2.set_xlabel('Freq (Hz)')
    ax2.set_ylabel('|Y(freq)|')
    plt.show()

def fft(t, data):
    """
    Computes FFT of some given function.
    Inputs: 
        t (time as list), 
        data (signal data as list)
    Returns: 
        frq (one side frequency range of signal), 
        Y (fft of signal)
    """
    dt = t[-1] / len(t)
    Fs = 1/dt
    n = len(data) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    Y = np.fft.fft(data)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    return frq,Y

def plot_maf(t, data, data_maf, n, title = None):
    if title is not None:
        plt.title(f"{title} MAF averaged over {n} samples", fontsize=16)
    plt.plot(t,data,'k')
    plt.plot(t,data_maf,'r')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def maf(t, data, n):
    """
    Performs moving average filter over some data and returns smoothed data.
    Inputs:
        t (time as list), 
        data (signal data as list)
        n (number of datapoints to average on)
    Returns: 
        list of smoothed data with delay as leading zeros
    """
    data_maf = [0]*(n-1)
    for idx,_ in enumerate(data[n-1::]):
        window = data[idx:idx+n-1]
        data_maf.append(np.average(window))
    return data_maf

def plot_iir(t, data, data_iir, A, title = None):
    if title is not None:
        plt.title(f"{title} IIR smoothed with A={A}, B={1-A}", fontsize=16)
    plt.plot(t,data,'k')
    plt.plot(t,data_iir,'r')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def iir(t, data, A):
    """
    Performs infinite impulse filter over some data and returns smoothed data.
    Inputs:
        t (time as list), 
        data (signal data as list)
        A (weight of new datapoint <1)
    Returns: 
        list of smoothed data
    """
    assert(0<=A<=1, f"expected A in [0,1], got: {A}")
    B = 1 - A
    data_iir = data[0:1:]
    running_average = data[0]
    for datapoint in data[1::]:
        data_iir.append( A * datapoint + B * running_average)
        running_average = data_iir[-1]
    return data_iir

def main():
    """
    Main Loop
    """
    for filename in ['sigA.csv','sigB.csv','sigC.csv','sigD.csv']:
        t = []
        data = []
        with open(filename) as f:
            # open the csv file
            reader = csv.reader(f)
            for row in reader:
                # read the rows 1 one by one
                t.append(float(row[0])) # leftmost column
                data.append(float(row[1])) # second column
        frq, Y = fft(t, data)
        plot_fft(t,data,frq,Y, title=filename[:-4])

        for moving_average in [5,10,25,50,100,250,500,1000]:
            data_maf = maf(t, data, moving_average)
            plot_maf(t, data, data_maf, moving_average, title=filename[:-4])

        for A in [0, 0.01, 0.1, 0.25, 0.5, 1]:
            data_iir = iir(t, data, A)
            plot_iir(t, data, data_iir, A, title=filename[:-4])

if __name__ == "__main__":
    exit(main())