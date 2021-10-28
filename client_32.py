import os
import time
import signal
import sys
import struct
import serial
import serial.tools.list_ports
from datetime import datetime
import board
import socket
from digitalio import DigitalInOut, Direction, Pull
from subprocess import Popen, DEVNULL, PIPE, run
import threading

debug = False

#MLFRA result LED light
mlfraResultLED = DigitalInOut(board.D12)
mlfraResultLED.direction = Direction.OUTPUT
mlfraResultLED.value = False

#MLFRA error LED light
mlfraErrorLED = DigitalInOut(board.D16)
mlfraErrorLED.direction = Direction.OUTPUT
mlfraErrorLED.value = False

#MLFRA running LED light
mlfraRunningLED = DigitalInOut(board.D21)
mlfraRunningLED.direction = Direction.OUTPUT
mlfraRunningLED.value = False

#Error output GPIO
mlfraError = DigitalInOut(board.D6)
mlfraError.direction = Direction.OUTPUT
mlfraError.value = True

#MLFRA results output GPIO
mlfraResult = DigitalInOut(board.D13)
mlfraResult.direction = Direction.OUTPUT
mlfraResult.value = False

#Last calibration button
lastCali = DigitalInOut(board.D20)
lastCali.direction = Direction.INPUT
lastCali.pull = Pull.DOWN

#Create and bind to the calibration serial port
caliSer = serial.Serial("/dev/ttyS0",  57600)

#Create and bind the data serial object
try:
    a = serial.tools.list_ports.comports()
    for p in a:
        if("Arduino" in p.manufacturer and p.serial_number == "85036313630351802031"):
            ser = serial.Serial(p.device, baudrate=115200)
            break
except Exception as err: 
    print("ERROR loading ADC arduino")
    print(err)
    sys.exit(1)

#The debugging file output
if(debug):
    theFilename = "/" + os.path.join("home","pi","Documents","MLFRA",datetime.now().strftime("%Y%m%d") + ".txt")
    if(not os.path.isfile(theFilename)):
        f = open(theFilename, 'w')
        f.write("current_time,mlfra_result,mlfra_percent,time\n")
        f.close()

def startMLServer():
    print("Starting MLFRA Server")
    cmd_str = "ds64-shell -c 'cd /home/pi/Documents/MLFRA; sudo python3 server_64.py'"
    proc = Popen([cmd_str], shell=True, 
                  stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)

def stopMLServer():
    #Query the virtual machine
    print("Stopping MLFRA Server")
    cmd_str = "ds64-shell -c 'ps -aux' | grep python3 | awk '{print $2}'"
    proc = run([cmd_str], shell=True, stderr=DEVNULL, stdout=PIPE)
    
    try:
        #Extract the PID to kill
        theVals = proc.stdout.decode('utf-8').strip("\n")
        firstVal = theVals.strip("\n").split("\n")

        for thePID in firstVal:
            #Kill the server by PID
            kproc = run(["ds64-shell -c 'sudo kill -1 " + str(thePID) + "'"], \
                         shell=True, stderr=PIPE, stdout=PIPE)
    except:
        print("Server not running, couldn't find PID")

def isServerAlive():
    cmd_str = "ds64-shell -c 'ps -aux' | grep python3 | awk '{print $2}'"
    proc = run([cmd_str], shell=True, stderr=DEVNULL, stdout=PIPE)

    try:
        #Extract the PID to kill
        theVals = proc.stdout.decode('utf-8').strip("\n")
        firstVal = theVals.strip("\n").split("\n")
        thePID = int(firstVal[0])

        if(thePID != None and thePID > 0):
            return True
        else:
            return False 
    except:
            return False

def queryMLFRA(theBlock, blockSlice):
    #Serve connection static variables
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server

    #Get the start time
    stime = time.monotonic()

    #Check to make sure server is alive
    if(not isServerAlive()):
        return -4

    #Set LED light to running
    mlfraRunningLED.value = True

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            #Convert the array to byte array
            arrangedArr = [None] * len(theBlock)
            arrangedInd = 0
            for i in range(blockSlice, len(theBlock)):
                arrangedArr[arrangedInd] = str(round(theBlock[i],2))
                arrangedInd += 1
            for i in range(0, blockSlice):
                arrangedArr[arrangedInd] = str(round(theBlock[i],2))
                arrangedInd += 1
            dataSend = bytearray((','.join(arrangedArr)).encode('utf-8'))

            #send the byte array over the socket
            s.connect((HOST, PORT))
            s.sendall(dataSend)

            #Recieve the results
            data = b''
            while True:
                part = s.recv(1024)
                data += part
                if len(part) < 1024: break

            #Decode the results
            runRes = data.decode("utf-8").split(",")
            
            #Append the time results
            elapsedSec = round((time.monotonic() - stime), 2)
            runRes.append(elapsedSec)

            #Write the output results to the debugging file
            if(debug):
                f = open(theFilename, 'a')
                f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ",")
                f.write(str(runRes[0]) + "," + str(runRes[1]) + "," + str(runRes[2]) + "\n")
                f.close()

            #Print the output results and update the LED
            print(runRes) 
            mlfraRunningLED.value = False

            #update the output digital lines with the result
            if(runRes != None):
                intRes = int(runRes[0])
                if(intRes == 0):
                    #Return fluid non-responsive
                    mlfraResultLED.value = mlfraResult.value = False
                    mlfraError.value = mlfraErrorLED.value = False
                    return(0)
                elif(intRes == 1):
                    #Return fluid responsive
                    mlfraResultLED.value = mlfraResult.value = True
                    mlfraError.value = mlfraErrorLED.value = False
                    return(1)
                else:
                    #Return MLFRA error
                    mlfraResultLED.value = mlfraResult.value = False
                    mlfraError.value = mlfraErrorLED.value = True
                    return(intRes)
            else:
                return(-2)
    except Exception as err:
        mlfraResultLED.value = mlfraResult.value = False
        mlfraError.value = mlfraErrorLED.value = True
        mlfraRunningLED.value = False
        print(err)
        return(-3)

#Create signal handler for ending program to kill server
def signal_handler(sig, frame):
    mlfraResultLED.value = False
    mlfraResult.value = False
    mlfraError.value = True
    mlfraErrorLED.value = False
    mlfraRunningLED.value = False
    stopMLServer()
    ser.close()
    caliSer.close()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

#Stop all servers that are orphaned and start a fresh instance
stopMLServer()
startMLServer()
time.sleep(2)

#Baseline variables
checkInterval = 10 * 100 # 10 seconds at 100 hz
blockSize = 60 * 100 # 60 seconds at 100 hz
checkCali = 0.2 * 100 # Check for last cali every 0.2 seconds
curInd = 0
theData = [0] * blockSize

#The function for converting ADC to mmHg
linear_M = 0.0
linear_B = 0.0
calibrated = False
fxnFile = "/home/pi/Documents/MLFRA/fxn.txt"
if(not os.path.exists(fxnFile)):
    f = open(fxnFile, 'w')
    f.write("linear_M,linear_B\n")
    f.close()

#The start byte
startDataByte=b'\xff'

#The thread variable for keeping track of if its alive
theThread = None

#loop for forever
while True:
    try:
        #Break into calibration sequence
        if(caliSer.inWaiting() > 0):
            theVal = caliSer.readline()
            theStr = theVal.decode("utf-8")
            theIndicies = theStr.strip("/r").strip("/n").split(",")           
            linear_M = float(theIndicies[0]) / 100000.0
            linear_B = float(theIndicies[1]) / 1000.0
            f = open(fxnFile, 'a')
            f.write(str(linear_M) + "," + str(linear_B) + "\n")
            f.close()
            calibrated = True
        
        #Read the serial value (0xff followed by 16 bit signed integer)
        singleByte = ser.read(1)

        #if byte is equal to start byte
        if(singleByte == startDataByte):
            #Read in the next two bytes
            line = ser.read(2) 

            #Make sure that the most significant byte is not the start byte
            while(line[0] == startDataByte):
                newLine = bytes('us')
                newLine.append = line[1]
                newLine.append = ser.read(1)
                del line
                line = newLine
                del newLine
                print("Prevented Frame Shift")
        
            #Parse the value (signed 16 bit integer)
            parsedVal = int(struct.unpack('>h', line)[0])
       
            #Place the new value into the array 
            theData[curInd] = (parsedVal * linear_M) - linear_B
            #print(theData[curInd], parsedVal, linear_M, linear_B)
            curInd += 1

            #Check to see if it the calibration button is pressed to load old calibration settings
            if(not calibrated and curInd % checkCali == 0 and lastCali.value):
                #Load the last time of the file
                lastLine = None
                f = open(fxnFile)
                for ind, line in enumerate(f):
                    if(ind > 0 and len(line) > 1): lastLine = line.strip("\n")
                f.close()
               
                #Parse the last line into the two values and convert to flow 
                if(lastLine != None):
                    theVals = lastLine.split(",")
                    linear_M = float(theVals[0])
                    linear_B = float(theVals[1])
                    calibrated = True
                    print(linear_M, linear_B)

            #Query the MLFRA only if the last thread is dead
            if(calibrated and curInd % checkInterval == 0 and 
                (theThread == None or (theThread != None and not theThread.is_alive()))):
                #Make Copies of the data and send data in a separate thread            
                toSendCopy = theData.copy()
                toSendInd = curInd
                theThread = threading.Thread(target=queryMLFRA, args=(toSendCopy, toSendInd,))
                theThread.start()

            #Reset the pointer once at the end to the start
            if(curInd >= blockSize): curInd = 0
    except Exception as err:
        print(err) 

