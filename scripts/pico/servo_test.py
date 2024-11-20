from machine import Pin, PWM
from time import sleep

# SETUP
servo = PWM(Pin(0))
servo.freq(50)

# LOOP
for i in range(1000000, 2000000, 10000):
    servo.duty_ns(i)
    print(i)
    sleep(0.2)
for i in reversed(range(1000000, 2000000, 10000)):
    servo.duty_ns(i)
    print(i)
    sleep(0.2)
servo.duty_ns(1500000)
servo.deinit()

