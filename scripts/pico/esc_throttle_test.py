from machine import Pin, PWM, reset
from time import sleep

print("Hold tight! Your car may dash out!")
# SETUP
motor = PWM(Pin(15))
motor.freq(50)
sleep(3)

# LOOP
try:
    print("Speeding up going forward")
    for i in range(1500000, 1800000, 5000): # forward up
        motor.duty_ns(i)
        print(i)
        sleep(0.2)
    print("Slowing down")
    for i in reversed(range(1500000, 1800000, 5000)): # forward down
        motor.duty_ns(i)
        print(i)
        sleep(0.2)
    print("Speeding up going backward")
    for i in reversed(range(1200000, 1500000, 5000)): # forward up
        motor.duty_ns(i)
        print(i)
        sleep(0.2)
    print("Slowing down")
    for i in range(1200000, 1500000, 5000): # forward down
        motor.duty_ns(i)
        print(i)
        sleep(0.2)
    print("Calibrate ESC if necessary.")
except:
    print("Exception")
finally:
    motor.duty_ns(0)
    sleep(1)
    motor.deinit()
    reset()
    


