import time
from tinkerforge.ip_connection import IPConnection
from tinkerforge.brick_silent_stepper import BrickSilentStepper
from tinkerforge.bricklet_analog_in_v3 import BrickletAnalogInV3
from pynput import keyboard


class methods():
    """
    brick_lets_module:list = ["bricklet_analog_in_v3", "bricklet_analog_out_v3", "bricklet_color_v2", "brick_silent_stepper"]
    brick_lets_objects:list = ["BrickletAnalogInV3", "BrickletAnalogOutV3", "BrickletColorV2", "BrickSilentStepper"]
    brick_lets_names:list = ["analog_in", "analog_out", "colour", "stepper"]
    """

    
    @staticmethod
    def setup(UID_stepper, UID_analog_in):
        HOST = "localhost"
        PORT = 4223

        # Tinkerforge objects are created
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
    """
    @staticmethod
    def resolution_convert(resolution):
        match resolution:
            case 8:
                return 1
            case 7:
                return 2
            case 6:
                return 4
            case 5:
                return 8
            case 4:
                return 16
            case 3:
                return 32
            case 2:
                return 64
            case 1:
                return 128
            case 0:
                return 256
    """

    @staticmethod
    def gain_integration_time_convert(gain:int, integration_time:int):
        # this function acts like a dictionary and returns the decoded gain and integration time
        match gain:
            case 0:
                gain = 1
            case 1:
                gain = 4
            case 2:
                gain = 16
            case 3:
                gain = 60
            case _:
                input("The configured gain can not be converted.")
        match integration_time:
            case 0:
                integration_time = 2.4
            case 1:
                integration_time = 24
            case 2:
                integration_time = 101
            case 3:
                integration_time = 154
            case 4:
                integration_time = 700
            case _:
                input("The configured integration time can not be converted.")
        return gain, integration_time