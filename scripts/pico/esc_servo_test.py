from machine import Pin, PWM
from time import sleep

# SETUP
servo = PWM(Pin(16))
servo.freq(50)

# LOOP
try:
    print("Turning Left Fully")
    for i in range(1500000, 1900000, 10000): 
        servo.duty_ns(i)
        print(i)
        sleep(0.2)
    print("Turning Left to Center")
    for i in reversed(range(1500000, 1900000, 10000)): 
        servo.duty_ns(i)
        print(i)
        sleep(0.2)
    print("Turning Right Fully")
    for i in reversed(range(1100000, 1500000, 10000)): 
        servo.duty_ns(i)
        print(i)
        sleep(0.2)
    print("Turning Right to Center")
    for i in range(1100000, 1500000, 10000): 
        servo.duty_ns(i)
        print(i)
        sleep(0.2)
except:
    print("Exception")
finally:
    servo.duty_ns(0)
    servo.deinit()

