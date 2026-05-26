import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import constants
import locale
from scipy.optimize import curve_fit

from functions_for_eval import *

test_phase = True

def main(num, axs_all):
    # enable latex in plots
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams.update(mpl.rcParamsDefault)

    locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

    # uncertainty of the Dataframe variables
    uncertainty_frequency = 2 * constants.pi * 0.225 / (2 * np.sqrt(3))
    uncertainty_amplitude = 10 # mV

    mes_points = 100 # number of points per measurement

    # dictionaries that store all parameters of for each measurement
    # this should be replaced by one dictionary for better practice in the future
    fit_guesses = {
        3 : [100, 3.2, 0.2, 0],
        4 : [200, 3.3, 0.2, 0],
        5 : [200, 3.2, 0.3, 0],
        7 : [200, 3.2, 0.3, 0]
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

    mask_type = maskings[num][0]
    # 0 => no masking
    # 1 => mask only left side
    # 2 => mask only right side
    # 3 => mask both sides
    # 4 => middle

    lower_bound = maskings[num][1]
    upper_bound = maskings[num][2]



    # path is user specific and needs to be adjusted according to the path created in L2.Intermediate.py
    path = rf"C:\Versuchssoftware\GP1C\L2\freq_amp"
    file_name = rf"\Messung_{num}_curr_{curr}.csv"

    df = pd.read_csv(path + file_name, sep = ";") # load data into Dataframe

    # create subplots for plots and monte carlo histograms
    fig, axs = plt.subplots(tight_layout = True)
    fig_hist, axs_hist = plt.subplots(3, 1, tight_layout = True)

    frequency = list(df["frequency"])
    amplitude = list(df["amplitude max"])

    kwargs = {
        "capsize" : 3, 
        "capthick" : 0.3,
        "alpha" : 0.7,
        "elinewidth" : 0.3
    }


    match mask_type:
        case 0:
            qf, u_qf, curr = evaluation.eval_L2(frequency, amplitude, axs, axs_all, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, axs_hist, num, curr, test_phase)
            axs.errorbar(frequency, amplitude, xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", c = "orange", **kwargs)
        case 1:
            qf, u_qf, curr = evaluation.eval_L2(frequency[lower_bound:], amplitude[lower_bound:], axs, axs_all, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, axs_hist, num, curr, test_phase)
            axs.errorbar(frequency[lower_bound:], amplitude[lower_bound:], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", c = "orange", **kwargs)
            axs.errorbar(frequency[:- (mes_points - lower_bound)], amplitude[:- (mes_points - lower_bound)], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", c = "m", **kwargs)
        case 2:
            qf, u_qf, curr = evaluation.eval_L2(frequency[:upper_bound], amplitude[:upper_bound], axs, axs_all, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, axs_hist, num, curr, test_phase)
            axs.errorbar(frequency[:upper_bound], amplitude[:upper_bound], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", c = "orange", **kwargs)
            axs.errorbar(frequency[upper_bound:], amplitude[upper_bound:], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", c = "m", **kwargs)
        case 3:
            masked_data_freq = list(frequency.copy())
            masked_data_amp = list(amplitude.copy())
            qf, u_qf, curr = evaluation.eval_L2(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], axs, axs_all, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, axs_hist, num, curr, test_phase)
            axs.errorbar(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", c = "orange", **kwargs)
            del masked_data_freq[lower_bound:upper_bound]
            del masked_data_amp[lower_bound:upper_bound]
            axs.errorbar(masked_data_freq, masked_data_amp, xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", c = "m", **kwargs)
        case 4:
            masked_data_freq = list(frequency.copy())
            masked_data_amp = list(amplitude.copy())
            del masked_data_freq[lower_bound:upper_bound]
            del masked_data_amp[lower_bound:upper_bound]
            qf, u_qf, curr = evaluation.eval_L2(masked_data_freq, masked_data_amp, axs, axs_all, uncertainty_frequency, uncertainty_amplitude, active_fit_guess, num, curr, test_phase)
            axs.errorbar(masked_data_freq, masked_data_amp, xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"gefittete Messdaten", c = "orange", **kwargs)
            axs.errorbar(frequency[lower_bound:upper_bound], amplitude[lower_bound:upper_bound], xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = f"maskierte Messdaten", c = "m", **kwargs)
        case _:
            qf = u_qf = 0

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
    axs.set_title(rf'Messung bei Stromstärke {curr} A')
    if test_phase == False: # save figure
        fig.savefig(fname = rf"Messung_{num}_current_{curr}_v1.pdf", format = "pdf")
        fig_hist.savefig(fname = rf"Monte_Carlo_Verteilung_{num}_current_{curr}_v1.pdf", format = "pdf")

    return qf, u_qf, curr

if __name__ == "__main__":
    qf_li = []
    u_qf_li = []
    curr_li = []

    fig_all, axs_all = plt.subplots(tight_layout = True)

    for i in [4,5,7]:
        qf, u_qf, curr = main(i, axs_all)
        qf_li.append(qf)
        u_qf_li.append(u_qf)
        curr_li.append(curr)

    def quadratic(x, a, b, c):
        return 1 / (a * np.square(x) + b * x + c)

    fig, axs = plt.subplots()

    popt, pcov = curve_fit(quadratic, curr_li, qf_li)

    x = np.linspace(0.25,0.75, 10000)

    y = [quadratic(i, popt[0], popt[1], popt[2]) for i in x]

    axs.errorbar(curr_li, qf_li, yerr = u_qf_li, capsize = 3, fmt = "o")
    axs.plot(x,y)

    axs.grid()
    axs.set_xlabel(r'Stromstärke $I$ [A]')
    axs.set_ylabel(r'Gütefaktor $Q$ []')
    axs.set_title(rf'$I$-$Q$-Diagramm für alle Messungen')

    axs_all.legend()
    axs_all.grid()
    axs_all.set_xlabel(r'Frequenz $f$ [$\text{s}^{-1}$]')
    axs_all.set_ylabel(r'Amplitude $A$ [mV]')
    axs_all.set_title(rf'Alle Messungen')

    if test_phase == False: # save figure
        fig.savefig(fname = rf"Q_I_Diagramm_v1.pdf", format = "pdf")
        fig_all.savefig(fname = rf"Poster_plot_v1.pdf", format = "pdf")

    plt.show()