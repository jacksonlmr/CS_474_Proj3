import math
import numpy as np
import matplotlib.pyplot as plt

def magnitude(dft: np.ndarray):
    mag_list = []
    for x in range(dft.size):
        real = dft[x].real
        imag = dft[x].imag
        mag = math.sqrt(real**2+imag**2)
        mag_list.append(mag)

    return np.array(mag_list)

def phase(dft: np.ndarray):
    phase_list = []
    for x in range(dft.size):
        real = dft[x].real
        imag = dft[x].imag
        phase = math.atan2(imag, real)
        phase_list.append(phase)

    return np.array(phase_list)


# def phase(dft):
#     phase_list = []
#     for real, imag in zip(dft.real, dft.imag):
#         phase = math.atan(imag/real)
#         phase_list.append(phase)

#     return phase_list

def center_spectrum(signal: np.ndarray):
    centered_signal = np.empty_like(signal)
    for x in range(signal.size):
        centered_signal[x] = ((-1)**x)*signal[x]

    return centered_signal

def create_figure(name: str, x: np.ndarray, y: np.ndarray, outfile_path: str, o: bool):
    if o:
        plt.figure(figsize=(10, 3))
        plt.plot(x, y, 'o')
        plt.title(name)
        plt.savefig(f"{outfile_path}\{name}.png")
    else:
        plt.figure(figsize=(10, 3))
        plt.plot(x, y)
        plt.title(name)
        plt.savefig(f"{outfile_path}\{name}.png")