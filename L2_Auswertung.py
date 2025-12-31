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

mes_points = 100

num = 5
curr = 0.5

lower_bound = 39
upper_bound = 0

fit_guesses = [500, 3.2, 0.3, 0, 0]

test_phase = True

mask_type = 1
# 0 => no masking
# 1 => mask only left side
# 2 => mask only right side
# 3 => mask both sides

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
    # path is user specific and needs to be adjusted according to the path created in L2.Intermediate.py
    path = rf"C:\Versuchssoftware\GP1C\L2\freq_amp"
    file_name = rf"\Messung_{num}_curr_{curr}.csv"

    df = pd.read_csv(path + file_name, sep = ";")

    fig, axs = plt.subplots(tight_layout = True)

    frequency = df["frequency"]
    amplitude = df["amplitude max"]
    
    match mask_type:
        case 0:
            evaluation.eval_L2(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], axs, uncertainty_frequency, uncertainty_amplitude, fit_guesses, test_phase)
            axs.errorbar(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
        case 1:
            evaluation.eval_L2(frequency[lower_bound:], amplitude[lower_bound:], axs, uncertainty_frequency, uncertainty_amplitude, fit_guesses, test_phase)
            axs.errorbar(frequency[lower_bound:], amplitude[lower_bound:], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
            axs.errorbar(frequency[:- (mes_points - lower_bound)], amplitude[:- (mes_points - lower_bound)], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", capsize = 3, c = "m", fmt = "o")
        case 2:
            evaluation.eval_L2(frequency[:upper_bound], amplitude[:upper_bound], axs, uncertainty_frequency, uncertainty_amplitude, fit_guesses, test_phase)
            axs.errorbar(frequency[:upper_bound], amplitude[:upper_bound], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
            axs.errorbar(frequency[upper_bound:], amplitude[upper_bound:], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", capsize = 3, c = "m", fmt = "o")
        case 3:
            masked_data_freq = list(frequency.copy())
            masked_data_amp = list(amplitude.copy())
            evaluation.eval_L2(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], axs, uncertainty_frequency, uncertainty_amplitude, fit_guesses, test_phase)
            axs.errorbar(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
            del masked_data_freq[lower_bound:upper_bound]
            del masked_data_amp[lower_bound:upper_bound]
            axs.errorbar(masked_data_freq, masked_data_amp, xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", capsize = 3, c = "m", fmt = "o")
    
    axs.legend()
    axs.grid()
    axs.set_xlabel(r'Frequenz $f$ [$\text{s}^{-1}$]')
    axs.set_ylabel(r'Amplitude $A$ [mV]')
    axs.set_title(rf'Messung {num} bei Stromstärke {curr} A')
    plt.show()
    if test_phase == False:
        fig.savefig(fname = rf"Messung_{num}_current_{curr}_v1.pdf", format = "pdf")

if __name__ == "__main__":
    main()