import pandas as pd
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
from os import makedirs, listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

def main(num):
    path = rf"C:\Versuchssoftware\GP1C\L2\waves\Messung_{num}\\"

    min_rotations_per_second:float = 0.1
    max_rotations_per_second:float = 1 # rotation per second
    measurements_per_vel:int = 100

    velocities = np.linspace(min_rotations_per_second, max_rotations_per_second, measurements_per_vel)
    frequencies = np.multiply(velocities, 2 * constants.pi)

    amplitude_fit:list = []
    amplitude_max:list = []

    def fit_func(t, A, omega, phi, B):
        return A * np.sin(omega * t + phi) + B

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    onlyfiles.sort()
        
    spli = onlyfiles[0].split("_")
    curr = spli[1].split(",")[0]

    for filee in onlyfiles:
        df = pd.read_csv(path + filee, sep = ";")
        spli = filee.split("_")
        vel = float(spli[2].split(".csv")[0])
        try:
            popt, pcov = curve_fit(fit_func, df["time"], df["amplitude"], p0 = [60, 2 * constants.pi * vel, 0, 24100], bounds = ([0, 0, 0, 2000],[10000, 4 * constants.pi * vel, 2 * constants.pi, 30000]))
        except:
            popt = ["Nan"]

        amplitude_fit.append(popt[0])

        amplitude_max.append(df["amplitude"].max() - 2460)

    data_tmp = {"frequency":frequencies, "amplitude fit":amplitude_fit, "amplitude max":amplitude_max}

    df_f_amp = pd.DataFrame(data_tmp)
    df_f_amp.to_csv(rf"C:\Versuchssoftware\GP1C\L2\freq_amp\Messung_{num}_curr_{curr}.csv", sep = ";")

    df_f_amp.plot()
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    for num in [3,4,5,7]:
        main(num)