"""
Upload this script to the Pico board, then rename it to main.py.
Receives PWM values via USB serial and outputs them to GPIO pins.
Handles emergency stop (END,END), timeout failsafe, and watchdog reset.
"""

import sys
import select
import time
from machine import Pin, PWM, WDT, reset

# PWM Setup
steering = PWM(Pin(0))
steering.freq(50)

throttle = PWM(Pin(15))
throttle.freq(50)

# Default safe PWM values (neutral)
center_pwm = 1415000  # Steering center
idle_pwm = 1500000    # Throttle stop

steering.duty_ns(center_pwm)
throttle.duty_ns(idle_pwm)

# Watchdog setup (1 second timeout)
wdt = WDT(timeout=1000)

# Track last valid input time
last_update = time.ticks_ms()

# Serial setup
poller = select.poll()
poller.register(sys.stdin, select.POLLIN)

try:
    time.sleep(3)
    print("Pico ready. Listening...")

    while True:
        wdt.feed()

        # Check if new data is available (non-blocking)
        if poller.poll(100):
            try:
                line = sys.stdin.readline().strip()
                buffer = line.split(',')

                if buffer[0] == "END" and buffer[1] == "END":
                    print("Shutdown signal received.")
                    throttle.duty_ns(idle_pwm)
                    steering.duty_ns(center_pwm)
                    break  # or use reset() if you want a full reboot

                if len(buffer) == 2:
                    ns_st = int(buffer[0])
                    ns_th = int(buffer[1])
                    steering.duty_ns(ns_st)
                    throttle.duty_ns(ns_th)
                    last_update = time.ticks_ms()
                    # print(f"PWM: {ns_st}, {ns_th}")  # optional debug

            except Exception as e:
                print("Input error:", e)

        # Failsafe: if no input for 500ms, go neutral
        if time.ticks_diff(time.ticks_ms(), last_update) > 500:
            throttle.duty_ns(idle_pwm)
            steering.duty_ns(center_pwm)
            # print("Failsafe: No input, reset PWM")  # optional debug

except Exception as e:
    print("Fatal error:", e)

finally:
    print("Pico shutting down")
    reset()
