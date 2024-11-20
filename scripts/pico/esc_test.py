from machine import Pin, PWM
from time import sleep

# SETUP
motor = PWM(Pin(15))
motor.freq(50)
sleep(3)

# LOOP
for i in range(1210000, 1800000, 10000): # forward up
    motor.duty_ns(i)
    print(i)
    sleep(0.2)
for i in reversed(range(1210000, 1800000, 10000)): # forward down
    motor.duty_ns(i)
    print(i)
    sleep(0.2)
for i in reversed(range(1090000, 1210000, 10000)): # reverse up
    motor.duty_ns(i)
    print(i)
    sleep(0.2)
for i in range(1090000, 1210000, 10000): # reverse down
    motor.duty_ns(i)
    print(i)
    sleep(0.2)
motor.duty_ns(1210000)
sleep(1)
motor.deinit()


