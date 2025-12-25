import pandas as pd
import numpy as np
import scipy.odr as odr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.differentiate import derivative
from scipy import constants


# Source - https://stackoverflow.com/a
# Posted by alko
# Retrieved 2025-12-25, License - CC BY-SA 3.0

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)


mpl.rcParams['text.usetex'] = True
mpl.rcParams.update(mpl.rcParamsDefault)

uncertainty_frequency = 2 * constants.pi * 0.225 / (2 * np.sqrt(3))
uncertainty_amplitude = 0.01 #10 mV

def u_A(params, u_params):
    result = 0
    for i in u_params:
        res = partial_derivative(fit_func, var = i, point = params)
        result += np.square(res.df[0] * u_params[i])
    return np.sqrt(result)

def fit_func(params:list):
    omega, a, omega_0, delta, B, C = params[0], params[1], params[2], params[3], params[4], params[5]
    return a / np.sqrt(np.square(np.square(omega_0) - np.square(omega + C)) + np.square(2 * delta * (omega + C))) + B # = A(\omega)

def fit_func_odr(params:list, omega):
    params.insert(0, omega)
    return fit_func(params)

def evaluation(frequency:list, amplitude:list, label, fig, axs):
    fit_func_odr = odr.Model(fit_func_odr)
    data = odr.RealData(frequency, amplitude, sx = uncertainty_frequency, sy = uncertainty_amplitude)
    odr_fit = odr.ODR(data,fit_func_odr, beta0 = [400, 3.3, 0.2, 0, 0])
    output = odr_fit.run()

    output.pprint()

    a, omega_0, delta, B, C = output.beta[0], output.beta[1], output.beta[2], output.beta[3], output.beta[4]
    u_a, u_omega_0, u_delta, u_B, u_C = output.sd_beta[0], output.sd_beta[1], output.sd_beta[2], output.sd_beta[3], output.sd_beta[4]

    x = np.linspace(min(frequency), max(frequency), 10000)
    y = [fit_func_odr([a, omega_0, delta, B, C], i) for i in x]
    axs.plot(x,y, label = f"ODR Fit", c = "cyan")

    peaks, props = find_peaks(y)
    resonance_frequency = x[peaks[0]]
    print(resonance_frequency)

    amplitude_res_freq = fit_func_odr([a, omega_0, delta, B, C], resonance_frequency)
    u_amplitude_res_freq = u_A(output.beta, [u_omega_0, u_a, u_delta, u_B, u_C])

    axs.errorbar(resonance_frequency, amplitude_res_freq, xerr = u_omega_0, yerr = u_amplitude_res_freq, label = r"Resonanzfrequenz $\omega_0$",c = "blue", fmt = "o", capsize = 3)

    print(rf"The resonance frequency is: {resonance_frequency} {u'\u00b1'} {u_omega_0}")

    amplitude_cut_off_freq = 0.707 * amplitude_res_freq
    u_amplitude_cut_off_freq = 0.707 * u_amplitude_res_freq
    print(rf"Amplitude at res freq: {amplitude_res_freq} {u'\u00b1'} {u_amplitude_res_freq}")
    print(rf"Amplitude at cut off freq: {amplitude_cut_off_freq} {u'\u00b1'} {u_amplitude_cut_off_freq}")
    
    args = (a, omega_0, delta, B, C)
    func = lambda omega, a, omega_0, delta, B, C: fit_func_odr([a, omega_0, delta, B, C], omega) - amplitude_cut_off_freq
    cut_off_freq_0 = least_squares(func, x0 = 2.5, bounds = (0, resonance_frequency), args = args)["x"][0]
    cut_off_freq_1 = least_squares(func, x0 = resonance_frequency+2, bounds = (resonance_frequency, np.inf), args = args)["x"][0]

    print(f"The cut off frequencies are: {cut_off_freq_0} and {cut_off_freq_1}")

    axs.errorbar(cut_off_freq_0, amplitude_cut_off_freq, xerr = u_omega_0, yerr = u_amplitude_cut_off_freq, label = r"Linke Grenzfrequenz $f_{g,1}$", c = "green", fmt = "o", capsize = 3)
    axs.errorbar(cut_off_freq_1, amplitude_cut_off_freq, xerr = u_omega_0, yerr = u_amplitude_cut_off_freq, label = r"Rechte Grenzfrequenz $f_{g,2}$", c = "purple", fmt = "o", capsize = 3)

    bandwidth = cut_off_freq_1 - cut_off_freq_0
    u_bandwidth = np.sqrt(2 * np.square(u_omega_0))
    print(rf"The bandwidth is equal to {bandwidth} {u'\u00b1'} {u_bandwidth}")

    quality_factor = resonance_frequency / bandwidth
    u_quality_factor = np.sqrt(np.square(u_omega_0 / bandwidth) + np.square(- resonance_frequency * u_bandwidth / np.square(bandwidth)))
    print(rf"The quality factor is equal to {quality_factor} {u'\u00b1'} {u_quality_factor}")

    decay_rate = resonance_frequency / (2 * quality_factor)
    print(f"The decay rate is equal to: {decay_rate}")


def main():
    path = rf"C:\Versuchssoftware\GP1C\L2\freq_amp"
    num = 4
    curr = 0.25
    
    fig, axs = plt.subplots(tight_layout = True)

    file_name = rf"\Messung_{num}_curr_{curr}.csv"

    df = pd.read_csv(path + file_name, sep = ";")

    frequency = df["frequency"]
    amplitude = df["amplitude max"]
    evaluation(frequency[10:-10], amplitude[10:-10], curr, fig, axs)
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