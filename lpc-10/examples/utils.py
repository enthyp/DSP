from scipy.signal import butter, lfilter


def butter_lowpass_filter(data, cutoff_freq, sample_rate=44100, order=5):
    nyq = 0.5 * sample_rate
    cutoff = cutoff_freq / nyq
    b, a = butter(order, cutoff, btype='lowpass')
    y = lfilter(b, a, data)
    return y