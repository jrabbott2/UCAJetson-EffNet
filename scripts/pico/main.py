"""
Upload this script to the pico board, then rename it to main.py.
Read dutycycle in nanoseconds via USB BUS.
"""
import sys
import select
from time import sleep
from machine import Pin, PWM, reset

# SETUP
steering = PWM(Pin(0))
steering.freq(50)
throttle = PWM(Pin(15))
throttle.freq(50)


# LOOP
try:
    sleep(3)  # get pico ready
    print("I'm listening...")
    poller = select.poll()
    poller.register(sys.stdin, select.POLLIN)
    event = poller.poll()

    while True:        
        for msg, _ in event:
            buffer = msg.readline().rstrip().split(',')
            # print(buffer) # debug
            # print(len(buffer)) # debug
            if len(buffer) == 2:
                try:
                    ns_st, ns_th = int(buffer[0]), int(buffer[1])
                    # if ns_st == ns_th == 'END':
                    #     break
                except ValueError:
                    pass
                # print(ns_st, ns_th) # debug
                steering.duty_ns(ns_st)
                throttle.duty_ns(ns_th)

except:
    pass
finally:
    print('Pico reset')
    reset()