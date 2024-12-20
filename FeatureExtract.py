import numpy as np
from scipy.fft import rfft, rfftfreq

def calc_PSD(freq_dist, N):
    PSD_list = []
    for key in freq_dist.keys():
        P = (freq_dist[key]**2)/N
        PSD_list.append(P)
    return PSD_list

def calc_PSE(p_list):
    temp_sum = 0
    for p in p_list:
        temp_sum += p*np.log(p)
    PSE = -temp_sum
    return PSE

def calc_spectral_entropy(data):
    SAMPLE_RATE = 10000

    time = np.array(range(len(data)))
    N = len(time)
    xf = rfftfreq(N,1/SAMPLE_RATE)
    yf = rfft(data)
    amp = np.abs(yf)

    freq_dist = {}
    for i, freq in enumerate(xf):
        freq_dist[freq] = amp[i]
    PSD_list = calc_PSD(freq_dist, N)
    p_list = PSD_list/sum(PSD_list)
    entropy = calc_PSE(p_list)
    return entropy

def extract_features(data):
    min = np.min(data)
    max = np.max(data)
    mean = np.mean(data)
    std = np.std(data)
    ave = np.mean(data)/len(data)
    spectral_entropy = calc_spectral_entropy(data)

    return [min, max, mean, ave, std, spectral_entropy]
