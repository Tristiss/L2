import time
from tinkerforge.ip_connection import IPConnection
from tinkerforge.brick_silent_stepper import BrickSilentStepper
from tinkerforge.bricklet_analog_in_v3 import BrickletAnalogInV3
from pynput import keyboard


class methods():
    @staticmethod
    def setup(UID_stepper, UID_analog_in):
        # sets up IP connections to the stepper and analog in bricklet and sets default configurations
        HOST = "localhost"
        PORT = 4223

        # Create Tinkerforge objects
        ipcon = IPConnection()

        stepper = BrickSilentStepper(UID_stepper, ipcon)
        analog_in = BrickletAnalogInV3(UID_analog_in, ipcon)

        ipcon.connect(HOST, PORT)
        # Don't use device before ipcon is connected

        stepper.set_motor_current(800) # 800 mA
        stepper.set_step_configuration(stepper.STEP_RESOLUTION_8, True) # 1/8 steps (interpolated)
        stepper.set_max_velocity(3000) # Velocity 3000 steps/s

        stepper.set_speed_ramping(200, 500) # the steppers decelleration is high to optimize the calibration

        stepper.enable() # Enable motor power
        return ipcon, stepper, analog_in
    
    @staticmethod
    def shut_down(stepper,ipcon): # Stop motor before disabling motor power
        # shuts down all necessary tinkerforge modules and disconnects the IP Connection
        stepper.stop() # Request motor stop
        methods.wait(stepper)
        stepper.disable() # Disable motor power
        ipcon.disconnect() # disconnect IP Connection

    @staticmethod
    def wait(stepper):
        # simple function that blocks the thread and waits for the stepper to stop moving
        while stepper.get_current_velocity() != 0:
            time.sleep(0.1)

    @staticmethod
    def monitor(flag):
        # this function is a daemon thread that raises a flag that the measurment and calibration loops look out for
        pressed_keys = set()
        def on_press(key):
            # checks if the pressed key is the Escape button
            pressed_keys.add(key)
            if keyboard.Key.esc in pressed_keys:
                print("Esc detected! Stopping measurement...")
                flag.set()  # Signal measurement thread to stop

        def on_release(key):
            # released keys are being deleted from the set of pressed keys
            pressed_keys.discard(key)

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()  # Blocks until shutdown