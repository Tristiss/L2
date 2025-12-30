import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import constants

from functions_for_eval import *

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update(mpl.rcParamsDefault)

uncertainty_frequency = 2 * constants.pi * 0.225 / (2 * np.sqrt(3))
uncertainty_amplitude = 0.01 #10 mV


"""
def partial_derivative(var:int, point:list):
    def dif(x:float, params:list):
        params = np.insert(params, 0, x)
        print(params)
        return fit_func(params)
    varia = point[var]
    return derivative(dif, varia, args = [point.tolist()])
"""

def main():
    path = rf"C:\Versuchssoftware\GP1C\L2\freq_amp"
    num = 4
    curr = 0.25
    
    fig, axs = plt.subplots(tight_layout = True)

    file_name = rf"\Messung_{num}_curr_{curr}.csv"

    df = pd.read_csv(path + file_name, sep = ";")

    frequency = df["frequency"]
    amplitude = df["amplitude max"]
    evaluation.eval_L2(frequency[10:-10], amplitude[10:-10], curr, fig, axs, uncertainty_frequency, uncertainty_amplitude)
    axs.errorbar(frequency[10:-10], amplitude[10:-10], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
    masked_data_freq = list(frequency.copy())
    masked_data_amp = list(amplitude.copy())
    del masked_data_freq[10:-10]
    del masked_data_amp[10:-10]
    axs.errorbar(masked_data_freq, masked_data_amp, xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", capsize = 3, c = "m", fmt = "o")
    
    axs.legend()
    axs.grid()
    axs.set_xlabel(r'Frequenz $f$ [$\text{s}^{-1}$]')
    axs.set_ylabel(r'Amplitude $A$ [mV]')
    axs.set_title(rf'Messung {num} bei Stromstärke {curr} A')
    plt.show()
    #fig.savefig(fname = rf"Messung_{num}_current_{curr}_v1.pdf", format = "pdf")

if __name__ == "__main__":
    main()