import numpy as np
from scipy.fft import rfft, rfftfreq

def calc_min(data):
    return np.min(data)

def calc_max(data):
    return np.max(data)

def calc_mean(data):
    return np.mean(data)

def calc_ave(data):
    mean_val = calc_mean(data)
    return mean_val/len(data)

def calc_std(data):
    return np.std(data)

def calc_PSD(freq_dist, N):
    PSD_list = []
    for key in freq_dist.keys():
        P = (freq_dist[key]**2)/N
        PSD_list.append(P)
    return PSD_list

def normalize_PSD(PSD_list):
    p_list = []
    psd_sum = sum(PSD_list)
    p_list = PSD_list/psd_sum
    # for P in PSD_list:
    #     small_p = P/sum(PSD_list)
    #     p_list.append(small_p)
    return p_list

def calc_PSE(p_list):
    temp_sum = 0
    for p in p_list:
        temp_sum += p*np.log(p)
    PSE = -temp_sum
    return PSE

def calc_spectral_entropy(data, arduino_delay=50e-3):
    SAMPLE_RATE = 10000

    time = np.array(range(len(data)))
    N = len(time)
    # print(21)
    xf = rfftfreq(N,1/SAMPLE_RATE)
    # print(22)
    yf = rfft(data)
    # print(23)
    amp = np.abs(yf)
    # print(24)
    #
    # print(enumerate(xf))

    freq_dist = {}
    # print(21)
    for i, freq in enumerate(xf):
        freq_dist[freq] = amp[i]
    # print(22)
    PSD_list = calc_PSD(freq_dist, N)
    # print(23)
    p_list = normalize_PSD(PSD_list)
    # print(24)
    entropy = calc_PSE(p_list)
    return entropy

def extract_features(data, image=False):
    # print(11)
    min = calc_min(data)
    # print(12)
    max = calc_max(data)
    # print(13)
    mean = calc_mean(data)
    # print(14)
    std = calc_std(data)
    # print(15)
    if image:
        ave = calc_ave_img(data)
        spectral_entropy = calc_spectral_entropy_img(data)
    else:
        ave = calc_ave(data)
        spectral_entropy = calc_spectral_entropy(data)

    return [min, max, mean, ave, std, spectral_entropy]

def calc_ave_img(img):
    mean_val = calc_mean(img)
    return mean_val / 255.0

def calc_spectral_entropy_img(img):
    hist = np.histogram(img, bins = 256, range=(0, 255))[0]
    hist = hist / (img.size + 1e-7)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy