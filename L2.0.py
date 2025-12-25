import time
import pandas as pd
import numpy as np
import threading
import matplotlib.pyplot as plt
import serial
import os.path
from scipy import constants
from scipy.optimize import curve_fit
from os import makedirs, listdir
from os.path import isfile, join

from Static_methods_L2 import *

# UID for the specific hardware
UID_stepper:str = "5VFLBs"
UID_analog_in:str = "F6U"

amplitude_data:dict = {}
path = r"C:\Versuchssoftware\GP1C\L2"

resolution:int = 8
min_rotations_per_second:float = 0.1
max_rotations_per_second:float = 1 # rotation per second
measurements_per_vel:int = 100

"""
min_current:float = 0
max_current:float = 1
measurements_per_current = 4
"""

current = 0.75

rotations_per_amp_mes = 6

# constants
zero_voltage:float = 2460 #mV
uncertainty_amplitude = 0.01 #10 mV

velocities = np.linspace(min_rotations_per_second, max_rotations_per_second, measurements_per_vel)
#currents = np.linspace(min_current, max_current, measurements_per_current)

frequencies = np.multiply(velocities, 2 * constants.pi)

steps_per_rotation:int = 200 * resolution

num = 7

"""
if isfile(path + rf"\data\L2-Messung_0.csv") == True:
    onlyfiles = [f for f in listdir(path + r"\data") if isfile(join(path + r"\data", f))]

    onlyfiles.sort()

    num = int(onlyfiles[-1].split("_")[1][0]) + 1

else:
    num = 0

print(num)
"""

flag = threading.Event()
t_monitor = threading.Thread(target = methods.monitor, args = (flag,), daemon = True) # create the monitoring thread
t_monitor.start() # initialise that thread

def fit_func(t, A, omega, phi, B):
    return A * np.sin(omega * t + phi) + B

def measurement(stepper, analog_in, vel):
    stepper.set_max_velocity(vel * steps_per_rotation)
    stepper.drive_forward()
    time.sleep(4 / vel) # wait for 4 rotations

    data = []
    times = []

    start = stepper.get_current_position()
    start_time = time.perf_counter()

    while stepper.get_current_position() - start < vel * steps_per_rotation * rotations_per_amp_mes and flag.is_set() != True:
        data.append(analog_in.get_voltage())
        times.append(time.perf_counter() - start_time)
    
    try:
        popt, pcov = curve_fit(fit_func, times, data, p0 = [60, 2 * constants.pi * vel, 0, 24100], bounds = ([0, 0, 0, 2000],[10000, 4 * constants.pi * vel, 2 * constants.pi, 30000]))
    except:
        popt = ["Nan"]
    
    amplitude = popt[0]
    print(popt)

    # Source - https://stackoverflow.com/questions/1274405/how-to-create-new-folder
    # Posted by mcandre, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-15, License - CC BY-SA 3.0
    #newpath = r'C:\Program Files\arbitrary' 
    #if not os.path.exists(newpath):
    #    os.makedirs(newpath)

    newpath = path + rf"\waves\Messung_{num}" 
    if not os.path.exists(newpath):
        makedirs(newpath)

    data_tmp = {"time":times, "amplitude":data}

    pd.DataFrame(data_tmp).to_csv(path + rf"\waves\Messung_{num}\wave-curr_{current},vel_{vel}.csv", sep = ";")

    print(vel)
    print(amplitude)

    return amplitude

def measurement_row(stepper, analog_in):
    amplitude_data.update({current:[]})
    for vel in velocities:
        amplitude = measurement(stepper, analog_in, vel)
        amplitude_data[current].append(amplitude)
        if flag.is_set() == True:
            raise Exception("User shut down the protocol")

def data_management():
    df = pd.DataFrame(amplitude_data, index = frequencies)

    df.to_csv(path + rf"\data\L2-Messung_{num}.csv", sep = ";")

    df.plot()
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    ipcon, stepper, analog_in = methods.setup(UID_stepper, UID_analog_in)

    #ser = serial.Serial("COM6", 9600)

    #ser.write(b"SOCP 0100\r")

    #ser.write(b"SOUT1\r")

    try:
        """for current in currents:
            if flag.is_set():
                break
            out = int(current * 100)
            msg = f"CURR 00{out:03d}\r" # set current not voltage
            print(msg)
            ser.write(msg.encode())
            time.sleep(0.5)"""

        measure = threading.Thread(target = measurement_row, args = (stepper, analog_in))
        measure.start()

        try:
            measure.join()
        except Exception as e:
            print(e)
            methods.shut_down(stepper, ipcon) 
            #ser.close()
            #break
    finally:
        methods.shut_down(stepper, ipcon)
        #ser.close()
        data_management()

if __name__ == "__main__":
    main()