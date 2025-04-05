from machine import Pin, PWM, reset
from time import sleep

print("Please calibrate your ESC follow the steps below:")
print("1. Unplug Pico")
print("2. Turn off ESC, hold (red, rectangle) 'set' button next to its switch, turn it back on until you heard repeating beeping sound, then release the 'set' button within 3 seconds")
print("3. Plug Pico back in, and run this MicroPython script.")
print("4. When prompt to set throttle neutral, press 'set' button. You'll hear one beep.")
print("5. When prompt to set throttle max forward, press 'set' button. You'll hear two beeps.")
print("6. When prompt to set throttle max reverse, press 'set' button. You'll hear three beeps.")

# SETUP
motor = PWM(Pin(15))
motor.freq(50)
motor.duty_ns(1500000)
sleep(0.5)
motor.duty_ns(1500000)
print("Please set thorttle neutral")
sleep(3)

# LOOP
try:
    motor.duty_ns(1800000)
    sleep(0.5)
    motor.duty_ns(1800000)
    print("Please set thorttle max forward")
    sleep(3)
    motor.duty_ns(1200000)
    sleep(0.5)
    motor.duty_ns(1200000)
    print("Please set thorttle max reverse")
    sleep(3)
    print("Heard 3 beeps? Your ESC is ready to run.")
except:
    print("Exception")
finally:
    motor.duty_ns(0)
    sleep(1)
    motor.deinit()
    reset()
    


