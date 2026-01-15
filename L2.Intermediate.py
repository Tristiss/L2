import pandas as pd
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

def main(num):
    path = rf"C:\Versuchssoftware\GP1C\L2\waves\Messung_{num}\\"

    # creating frequency array just like in L2.0.py because it is easier than creating it on the go 
    # by reading the filenames
    min_rotations_per_second:float = 0.1
    max_rotations_per_second:float = 1 # rotation per second
    measurements_per_vel:int = 100

    velocities = np.linspace(min_rotations_per_second, max_rotations_per_second, measurements_per_vel)
    frequencies = np.multiply(velocities, 2 * constants.pi)

    # empty lists where data is added later
    amplitude_fit:list = []
    amplitude_max:list = []

    def fit_func(t, A, B, omega, C):
        # fit function for a sine fit
        return A * np.sin(omega * t) + B * np.cos(omega * t) + C

    # get all files in the directory
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    onlyfiles.sort() # sort them so they line up with the frequency array
    #(isn't checked for but can be assumed)
        
    # get the current from one of the filenames
    spli = onlyfiles[0].split("_")
    curr = spli[1].split(",")[0]

    for filee in onlyfiles:
        # loads the file into a dataframe
        df = pd.read_csv(path + filee, sep = ";")
        spli = filee.split("_")
        # get the velocity from the filename
        vel = float(spli[2].split(".csv")[0])

        # try fitting the data against sine function add Nan when fit fails
        try:
            popt, pcov = curve_fit(fit_func, df["time"], df["amplitude"], p0 = [60, 60, 2 * constants.pi * vel, 24100], bounds = ([0, 0, 0, 0, 2000],[10000, 10000, 4 * constants.pi * vel, 30000]))
            amplitude = np.sqrt(np.square(popt[0]) + np.square(popt[1]))
        except:
            amplitude = "Nan"

        # append fit amplitude to list
        amplitude_fit.append(amplitude)

        # calculate the amplitude through the global maximum and subtract the measured zero 
        # (voltage at zero degree)
        amplitude_max.append(df["amplitude"].max() - 2460)

    # write data to csv and plot both amplitude colums
    data_tmp = {"frequency":frequencies, "amplitude fit":amplitude_fit, "amplitude max":amplitude_max}

    df_f_amp = pd.DataFrame(data_tmp)
    df_f_amp.to_csv(rf"C:\Versuchssoftware\GP1C\L2\freq_amp\Messung_{num}_curr_{curr}_v2.csv", sep = ";")

    df_f_amp.plot()
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # perform main function for each measurement 
    for num in [3,4,5,7]:
        main(num)