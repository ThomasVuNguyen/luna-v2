import wiringpi
import time

# Initialize wiringPi
wiringpi.wiringPiSetup()

# Pin 8 on the 26-pin header corresponds to GPIO1_D0 
# In the WiringPi numbering scheme, we need to find the correct wPi number
# According to the manual's GPIO readall output, physical pin 8 has wPi number 8

PIN = 3  # wPi number for GPIO1_D0 (physical pin 8)

# Set pin as INPUT with pull-up
wiringpi.pinMode(PIN, wiringpi.INPUT)
wiringpi.pullUpDnControl(PIN, wiringpi.PUD_UP)

print("Monitoring pin D0 (physical pin 8). Press Ctrl+C to exit.")

try:
    while True:
        # Read the current state of the pin
        state = wiringpi.digitalRead(PIN)
        
        if state == 0:  # LOW
            print("Pin is connected to ground")
        else:  # HIGH
            pass
            # print("Pin is NOT connected to ground")
        
        # Wait before checking again
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nExiting program")