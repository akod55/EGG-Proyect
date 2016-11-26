import numpy as np
import math

def HarrTransformation(frequency):
    a_i = []
    d_i = []

    for i in xrange(len(frequency)/2):
        # print frequency[i]
        a_i.append((frequency[2*i] + frequency[2*i+1]) /math.sqrt(2))
        d_i.append(((frequency[2*i] - frequency[2*i+1])/2.0)*math.sqrt(2))
    return a_i, d_i


def signal_energy(list_frequency):
    return sum([(x**2) for x in list_frequency])


def fourier_transform(function, frequency_rate):
    n = len(function)  # length of the signal
    k = np.arange(n)
    T = float(n) / float(frequency_rate)
    frq = k / T  # two sides frequency range

    # frq = frq[range(n / 2)]  # one side frequency range

    Y_real = np.fft.fft(function) / n  # fft computing and normalization
    Y = Y_real[range(n / 2)]

    return frq[range(n / 2)], Y


def fourier_trasform_song(song_data, frequency_rate):
    n = len(song_data)  # length of the signal
    k = np.arange(n)
    T = float(n) / float(frequency_rate)
    frq = k / T  # two sides frequency range
    frq = frq[range(n / 2)]  # one side frequency range

    normalized_song = [(x / 2 ** 8.) * 2 - 1 for x in song_data]  # 8-bit track , b is normalized

    Y_fourier = np.fft.fft(normalized_song) # fft computing and normalization
    real_fourier_part = len(Y_fourier) / 2

    return frq[:len(frq)-1], Y_fourier[:(real_fourier_part-1)]
