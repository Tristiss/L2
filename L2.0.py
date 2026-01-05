import time
import pandas as pd
import numpy as np
import threading
import os.path
from scipy import constants
from os import makedirs
import sys

from Static_methods_L2 import *

# UID for the specific hardware
UID_stepper:str = "5VFLBs"
UID_analog_in:str = "F6U"

path = r"C:\Versuchssoftware\GP1C\L2"

# experimental parameters
resolution:int = 8
min_rotations_per_second:float = 0.1 # rotation per second
max_rotations_per_second:float = 1 # rotation per second
measurements_per_vel:int = 100

# labeling
num = 7
current = 0.75 # A

rotations_per_amp_mes = 6 # number of rotations that are measured per velocity

# constants
zero_voltage:float = 2460 # mV
uncertainty_amplitude = 10 # mV

# array of all set velocities and frequencies
velocities = np.linspace(min_rotations_per_second, max_rotations_per_second, measurements_per_vel)
frequencies = np.multiply(velocities, 2 * constants.pi)

steps_per_rotation:int = 200 * resolution # steps per rotation (depends on the resolution of the stepper)

flag = threading.Event() # global variable that can be seen and changed by all active threads

# create the monitoring thread that changes the raises the flag when "Esc" is pressed
t_monitor = threading.Thread(target = methods.monitor, args = (flag,), daemon = True)
t_monitor.start()

def measurement(stepper, analog_in, vel):
    stepper.set_max_velocity(vel * steps_per_rotation)
    stepper.drive_forward()
    time.sleep(4 / vel) # wait for 4 rotations

    # temporary data storage
    data = []
    times = []

    # zero position and time
    start = stepper.get_current_position()
    start_time = time.perf_counter()

    while stepper.get_current_position() - start < vel * steps_per_rotation * rotations_per_amp_mes and flag.is_set() != True:
        # loop conditions stepper position not yet at rotations_per_amp_mes and flag not raised
        data.append(analog_in.get_voltage()) # measure the voltage / angle of the potentiometer
        times.append(time.perf_counter() - start_time) # log time

    # Source - https://stackoverflow.com/questions/1274405/how-to-create-new-folder
    # Posted by mcandre, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-15, License - CC BY-SA 3.0
    #newpath = r'C:\Program Files\arbitrary' 
    #if not os.path.exists(newpath):
    #    os.makedirs(newpath)

    # check wether or not the path to the raw measurement data exists
    newpath = path + rf"\waves\Messung_{num}" 
    if not os.path.exists(newpath):
        makedirs(newpath)

    # write data to csv file
    data_tmp = {"time":times, "amplitude":data}
    pd.DataFrame(data_tmp).to_csv(path + rf"\waves\Messung_{num}\wave-curr_{current},vel_{vel}.csv", sep = ";")

def measurement_row(stepper, analog_in):
    for vel in velocities:
        # perform a measurement for every velocity / frequency
        measurement(stepper, analog_in, vel)
        if flag.is_set() == True:
            raise Exception("User shut down the protocol")

def main():
    # setup any Hardware connections
    ipcon, stepper, analog_in = methods.setup(UID_stepper, UID_analog_in)

    # try to run the measurement, shut down the Hardware connections properly if either the flag gets
    # raised or an error accurs
    try:
        measure = threading.Thread(target = measurement_row, args = (stepper, analog_in))
        measure.start()

        try:
            measure.join()
        except Exception as e:
            print(e) # print error message
            methods.shut_down(stepper, ipcon) # initiate shut down protocol
            sys.exit() # exit program
    finally:
        methods.shut_down(stepper, ipcon) # shut down Hardware if measurement finished normally

if __name__ == "__main__":
    main()