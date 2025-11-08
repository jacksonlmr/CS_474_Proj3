import math
import numpy as np

def magnitude(dft):
    mag_list = []
    for real, imag in zip(dft.real, dft.imag):
        mag = math.sqrt(real**2+imag**2)
        mag_list.append(mag)

    return mag_list

def phase(dft):
    phase_list = []
    for real, imag in zip(dft.real, dft.imag):
        phase = math.atan(imag/real)
        phase_list.append(phase)

    return phase_list

def center_spectrum(signal: np.ndarray):
    centered_signal = np.empty_like(signal)
    for x in range(signal.size):
        centered_signal[x] = ((-1)**x)*signal[x]

    return centered_signal