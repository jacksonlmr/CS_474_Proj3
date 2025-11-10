import math
import numpy as np
import matplotlib.pyplot as plt
import os

def magnitude(dft: np.ndarray):
    mag_list = []
    for x in range(dft.size):
        real = dft[x].real
        imag = dft[x].imag
        mag = math.sqrt(real**2+imag**2)
        mag_list.append(mag)

    return np.array(mag_list)

def magnitude_2D(dft2D: np.ndarray):
    mag_array = np.zeros_like(dft2D, float)
    for row in range(dft2D.shape[0]):
        for col in range(dft2D.shape[1]):
            real = dft2D[row, col].real
            imag = dft2D[row, col].imag
            mag = math.sqrt(real**2+imag**2)
            mag_array[row, col] = mag

    return mag_array

def scale_magnitude(dft_mag: np.ndarray):
    scaled_mag = np.zeros_like(dft_mag)
    for row in range(dft_mag.shape[0]):
        for col in range(dft_mag.shape[1]):
            scaled_value = math.log(1 + dft_mag[row, col])
            scaled_mag[row, col] = scaled_value

    return scaled_mag

def phase(dft: np.ndarray):
    phase_list = []
    for x in range(dft.size):
        real = dft[x].real
        imag = dft[x].imag
        phase = math.atan2(imag, real)
        phase_list.append(phase)

    return np.array(phase_list)

def center_spectrum(signal: np.ndarray):
    centered_signal = np.empty_like(signal)
    for x in range(signal.size):
        centered_signal[x] = ((-1)**x)*signal[x]

    return centered_signal

def center_spectrum_2D(signal: np.ndarray):
    centered_signal = np.empty_like(signal)
    for row in range(signal.shape[0]):
        for col in range(signal.shape[1]):
            centered_signal[row, col] = ((-1)**(row+col))*signal[row, col]

    return centered_signal

def fft2D(Fuv: np.ndarray, isign):
    #loop through rows, do 1D fft
    Fuv_1D_rows = np.zeros_like(Fuv, dtype=np.complex128)

    for y in range(Fuv.shape[0]):#shape[0] is number of rows
        row = Fuv[y]
        fft_row = det_fft(row, isign)
        Fuv_1D_rows[y] = fft_row
    
        # print(Fuv_1D_rows[y])

    #loop through columns, do 1D fft
    Fuv_2D = np.zeros_like(Fuv, dtype=np.complex128)

    for x in range(Fuv_1D_rows.shape[1]):
        col = Fuv_1D_rows[:, x]
        # print(col)

        fft_col = det_fft(col, isign)
        Fuv_2D[x] = fft_col

    return Fuv_2D


def det_fft(array, isign):
    #backwards transform if positive, forward if negative
    if isign == 1:
        return np.fft.ifft(array, norm='forward')
    
    elif isign == -1:
        return np.fft.fft(array, norm='forward')
    
    else:
        raise ValueError('isign must be -1 or 1')

def generate_img(background_size: int, inner_size: int):
    background = np.zeros(shape=(background_size, background_size), dtype=np.uint8)
    interior = np.full(shape=(inner_size, inner_size), fill_value=255, dtype=np.uint8)

    #subtract inner from background
    #take 1/2 of the result, thats the index of the beginning

    inner_start = int((background_size-inner_size)//2)
    inner_end = inner_start + inner_size

    background[inner_start:inner_end, inner_start:inner_end] = interior

    return background


def create_figure(name: str, x: np.ndarray, y: np.ndarray, outfile_path: str, o: bool):
    # --- dark theme colors (high contrast) ---
    fig_bg   = '#0f1116'  # figure background
    ax_bg    = '#111827'  # axes background
    grid_col = '#374151'  # grid (dark gray)
    spine_col = '#9ca3af' # axes spines
    tick_col  = '#e5e7eb' # tick labels
    title_col = '#f3f4f6' # title/text
    line_col  = "#0bfc02" # line/stem color (light blue)
    dot_col   = '#0bfc02' # dot color (lighter blue)

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(ax_bg)

    if o:
        ax.plot(x, y, 'o', zorder=3, markersize=2, color=dot_col)
        ax.vlines(x, ymin=np.minimum(0, y), ymax=np.maximum(0, y),
                  linewidth=1, color=line_col)
    else:
        ax.plot(x, y, zorder=3, linewidth=1.4, color=line_col)

    # Desmos-like axes: ticks on the zero lines
    ax.spines['bottom'].set_position(('data', 0))   # x-axis at y=0
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['left'].set_position('zero')          # y-axis at x=0
    ax.yaxis.set_ticks_position('left')

    # Style spines/ticks/text for dark background
    for side in ['bottom', 'left', 'top', 'right']:
        ax.spines[side].set_color(spine_col)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(colors=tick_col, which='both')
    ax.set_title(name, color=title_col)

    # Subtle grid with good contrast
    ax.grid(True, linestyle=':', linewidth=0.8, color=grid_col, alpha=0.8)

    os.makedirs(outfile_path, exist_ok=True)
    fig.savefig(
        os.path.join(outfile_path, f"{name}.png"),
        bbox_inches='tight',
        dpi=150,
        facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def mapValues(input_img_array: np.ndarray):
    """
    Maps values from detected range to [0, 255]

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image

    **Returns**
    -----------
    >**output_img_array**: np.ndarray representing an image with the values mapped to [0, 255]
    """
    input_row, input_col = input_img_array.shape
    output_img_array = np.zeros((input_row, input_col), dtype=np.uint8)

    max_value = np.max(input_img_array)

    for current_row in range(input_row):
        for current_col in range(input_col):
            current_value = input_img_array[current_row, current_col]
            mapped_value = int(max(0, min(255, 255*(current_value/max_value))))
            output_img_array[current_row, current_col] = mapped_value

    return output_img_array