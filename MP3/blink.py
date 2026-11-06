from machine import Pin
from time import sleep

led = Pin("LED", Pin.OUT)

print("Blinking the LED 10 times...")
while True:
    led.toggle()
    sleep(1.0)

led.off()
print("Done.")
