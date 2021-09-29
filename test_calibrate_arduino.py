import time
import board
import busio
from digitalio import DigitalInOut, Direction, Pull
from adafruit_mcp3xxx.analog_in import AnalogIn
import adafruit_mcp4725

caliPinNum = board.D17
hiloPinNum = board.D27

#Bind to calibrate output pin
calibratePin = DigitalInOut(caliPinNum)
calibratePin.direction = Direction.OUTPUT
calibratePin.value = False

#Bind to highlow output pin
highVslowPin = DigitalInOut(hiloPinNum)
highVslowPin.direction = Direction.OUTPUT
highVslowPin.value = False

# Create and bind to the DAC
i2c = busio.I2C(board.SCL, board.SDA)
dac = adafruit_mcp4725.MCP4725(i2c)
dac.normalized_value = 0.0

#Set DAC value to almost zero
print("Setting DAC: Low")
dac.normalized_value = 0.1

#Trigger calibration low
highVslowPin.value = False
calibratePin.value = True
time.sleep(1)
calibratePin.value = False

#Set DAC back to zero
dac.normalized_value = 0.0

#Pause for chillin
time.sleep(3)

#Set DAC value to almost MAX
print("Setting DAC: High")
dac.normalized_value = 0.9

#Trigger calibration high
highVslowPin.value = True
calibratePin.value = True
time.sleep(1)
calibratePin.value = False

dac.normalized_value = 0.0

