import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from scipy.signal import butter, lfilter


def butter_lowpass_filter(data, cutoff_freq, sample_rate=44100, order=5):
    nyq = 0.5 * sample_rate
    cutoff = cutoff_freq / nyq
    b, a = butter(order, cutoff, btype='lowpass')
    y = lfilter(b, a, data)
    return y

def plot(data, labels, size=(20, 10)):
    fig, axes = plt.subplots(nrows=len(data), ncols=1, figsize=size)
    if len(data) == 1:
        axes = [axes]
    for samples, ax, label in zip(data, axes, labels):
        ax.plot(samples)
        ax.set_title(label)
      
    
def plot_audio(data, t_samples=10000, f_step=500, f_min=0, f_max=4000, fs=44100):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    # Find samples that determine frequency interval of interest.
    N = len(data)
    k_0, k_max = f_min * N / fs, f_max * N / fs  
    k_step = f_step * N / fs  
    
    amp_freq_response = np.abs(fft.rfft(data,n=10 * N))
    ax2.plot(amp_freq_response[:int(k_max)])
    ax2.set_title('Frequency domain')
    
    xticks = np.arange(k_0, k_max, k_step)
    xtick_labels = map(str, np.arange(f_min, f_max, f_step))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xtick_labels)
    
    ax1.plot(data[:t_samples])
    ax1.set_title('Time domain')