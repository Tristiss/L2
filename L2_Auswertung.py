import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import constants

from functions_for_eval import *

# enable latex in plots
mpl.rcParams['text.usetex'] = True
mpl.rcParams.update(mpl.rcParamsDefault)

# uncertainty of the Dataframe variables
uncertainty_frequency = 2 * constants.pi * 0.225 / (2 * np.sqrt(3))
uncertainty_amplitude = 10 # mV

mes_points = 100 # number of points per measurement

num = 7

# dictionaries that store all parameters of for each measurement
# this should be replaced by one dictionary for better practice in the future
fit_guesses = {
    3 : [100, 3.2, 0.2, 0, 0],
    4 : [200, 3.3, 0.2, 0, 0],
    5 : [200, 3.2, 0.3, 0],
    7 : [200, 3.2, 0.3, 0, 0]
}

horizontal_shift_dict = {
    3 : True,
    4 : True, 
    5 : False,
    7 : True
}

maskings = {
    3 : [3, 5, 47],
    4 : [0, 0, 0],
    5 : [1, 39, 0],
    7 : [0, 0, 0]
}

currents = {
    3 : 0,
    4 : 0.25,
    5 : 0.5,
    7 : 0.75
}

# get parameters for the current measurement evaluation
curr = currents[num]

active_fit_guess = fit_guesses[num]

test_phase = False

horizontal_shift = horizontal_shift_dict[num]

mask_type = maskings[num][0]
# 0 => no masking
# 1 => mask only left side
# 2 => mask only right side
# 3 => mask both sides
# 4 => middle

lower_bound = maskings[num][1]
upper_bound = maskings[num][2]


def main():
    # path is user specific and needs to be adjusted according to the path created in L2.Intermediate.py
    path = rf"C:\Versuchssoftware\GP1C\L2\freq_amp"
    file_name = rf"\Messung_{num}_curr_{curr}.csv"

    df = pd.read_csv(path + file_name, sep = ";") # load data into Dataframe

    # create subplots for plots and monte carlo histograms
    fig, axs = plt.subplots(tight_layout = True)
    fig_hist, axs_hist = plt.subplots(3, 1, tight_layout = True)

    frequency = df["frequency"]
    amplitude = df["amplitude max"]
    
    match mask_type:
        case 0:
            evaluation.eval_L2(frequency, amplitude, axs, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, horizontal_shift, axs_hist, test_phase)
            axs.errorbar(frequency, amplitude, xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
        case 1:
            evaluation.eval_L2(frequency[lower_bound:], amplitude[lower_bound:], axs, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, horizontal_shift, axs_hist, test_phase)
            axs.errorbar(frequency[lower_bound:], amplitude[lower_bound:], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
            axs.errorbar(frequency[:- (mes_points - lower_bound)], amplitude[:- (mes_points - lower_bound)], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", capsize = 3, c = "m", fmt = "o")
        case 2:
            evaluation.eval_L2(frequency[:upper_bound], amplitude[:upper_bound], axs, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, horizontal_shift, axs_hist, test_phase)
            axs.errorbar(frequency[:upper_bound], amplitude[:upper_bound], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
            axs.errorbar(frequency[upper_bound:], amplitude[upper_bound:], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", capsize = 3, c = "m", fmt = "o")
        case 3:
            masked_data_freq = list(frequency.copy())
            masked_data_amp = list(amplitude.copy())
            evaluation.eval_L2(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], axs, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, horizontal_shift, axs_hist, test_phase)
            axs.errorbar(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
            del masked_data_freq[lower_bound:upper_bound]
            del masked_data_amp[lower_bound:upper_bound]
            axs.errorbar(masked_data_freq, masked_data_amp, xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", capsize = 3, c = "m", fmt = "o")
        case 4:
            masked_data_freq = list(frequency.copy())
            masked_data_amp = list(amplitude.copy())
            del masked_data_freq[lower_bound:upper_bound]
            del masked_data_amp[lower_bound:upper_bound]
            evaluation.eval_L2(masked_data_freq, masked_data_amp, axs, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, horizontal_shift, test_phase)
            axs.errorbar(masked_data_freq, masked_data_amp, xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", capsize = 3, c = "orange")
            axs.errorbar(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", capsize = 3, c = "m", fmt = "o")
    
    for i in axs_hist: # edit style for histogram plots
        i.legend()
        i.grid()
        i.set_xlabel(r'Frequenz $f$ [$\text{s}^{-1}$]')
        i.set_ylabel(r'Anzahl')
    fig_hist.suptitle(r'Monte Carlo Verteilungen')
    
    # edit style for plots
    axs.legend()
    axs.grid()
    axs.set_xlabel(r'Frequenz $f$ [$\text{s}^{-1}$]')
    axs.set_ylabel(r'Amplitude $A$ [mV]')
    axs.set_title(rf'Messung {num} bei Stromstärke {curr} A')
    plt.show()
    if test_phase == False: # save figure
        fig.savefig(fname = rf"Messung_{num}_current_{curr}_v1.pdf", format = "pdf")

if __name__ == "__main__":
    main()