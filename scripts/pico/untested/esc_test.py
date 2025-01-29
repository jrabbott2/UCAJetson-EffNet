from machine import Pin, PWM
from time import sleep

# SETUP
motor = PWM(Pin(15))
servo = PWM(Pin(17))
motor.freq(50)
servo.freq(50)

# LOOP
for i in range(500000, 2500000, 5000):
    motor.duty_ns(i)
    servo.duty_ns(i)
    print(i)
    sleep(0.1)
for i in reversed(range(500000, 2500000, 5000)):
    motor.duty_ns(i)
    servo.duty_ns(i)
    print(i)
    sleep(0.1)
motor.duty_u16(0)
servo.duty_ns(0)
