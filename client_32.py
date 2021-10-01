import os
import time
import signal
import sys
import struct
import serial
from datetime import datetime
import board
import socket
from digitalio import DigitalInOut, Direction, Pull
from subprocess import Popen, DEVNULL, PIPE, run
import threading

#MLFRA result LED light
mlfraResultLED = DigitalInOut(board.D12)
mlfraResultLED.direction = Direction.OUTPUT
mlfraResultLED.value = False

#Running MLFRA light
mlfraErrorLED = DigitalInOut(board.D16)
mlfraErrorLED.direction = Direction.OUTPUT
mlfraErrorLED.value = False

#Error output GPIO
mlfraError = DigitalInOut(board.D6)
mlfraError.direction = Direction.OUTPUT
mlfraError.value = True

#MLFRA results output GPIO
mlfraResult = DigitalInOut(board.D13)
mlfraResult.direction = Direction.OUTPUT
mlfraResult.value = False

#Create and bind to the calibration serial port
caliSer = serial.Serial("/dev/ttyS0",  57600)

#Create and bind the data serial object
try:
    ser = serial.Serial("/dev/ttyACM0", baudrate=115200)
except: 
    ser = serial.serial("/dev/ttyACM1", baudrate=115200)

#The debugging file output
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
            f = open(theFilename, 'a')
            f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ",")
            f.write(str(runRes[0]) + "," + str(runRes[1]) + "," + str(runRes[2]) + "\n")
            f.close()

            #Print the output results and update the LED
            print(runRes) 

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
        print(err)
        return(-3)

#Create signal handler for ending program to kill server
def signal_handler(sig, frame):
    mlfraResultLED.value = False
    mlfraResult.value = False
    mlfraError.value = True
    mlfraErrorLED.value = False
    stopMLServer()
    ser.close()
    caliSer.close()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

#Stop all servers that are orphaned and start a fresh instance
stopMLServer()
startMLServer()
time.sleep(5)

checkInterval = 10 * 100 # 10 seconds at 100 hz
blockSize = 60 * 100 # 60 seconds at 100 hz
curInd = 0
theData = [0] * blockSize
linear_M = 0.0
linear_B = 0.0

startDataByte=b'\xff'
theThread = None

while True:
    try:
        #Break into calibration sequence
        if(caliSer.inWaiting() > 0):
            theVal = caliSer.readline()
            theStr = theVal.decode("utf-8")
            theIndicies = theStr.strip("/r").strip("/n").split(",")           
            linear_M = float(theIndicies[0]) / 10000.0
            linear_B = float(theIndicies[1]) / 100.0
        
        #Read the serial value (0xff followed by 16 bit signed integer)
        singleByte = ser.read(1)

        #if byte is equal to start byte
        if(singleByte == startDataByte):
            #Read in the next two bytes
            line = ser.read(2) 

            #Make sure that the most significant byte is not the start byte
            while(line[0] == startDataByte):
                line[0] = line[1]
                line[1] = ser.read(1)
                print("Prevented Frame Shift")
        
            #Parse the value and convert to float
            parsedVal = int(struct.unpack('>h', line)[0])
       
            #Place the new value into the array 
            theData[curInd] = (parsedVal * linear_M) - linear_B
            #print(theData[curInd], parsedVal, linear_M, linear_B)
            curInd += 1

            #Query the MLFRA only if the last thread is dead
            if(curInd % checkInterval == 0 and 
                (theThread == None or (theThread != None and not theThread.is_alive()))):
                #Make Copies of the data            
                toSendCopy = theData.copy()
                toSendInd = curInd
                theThread = threading.Thread(target=queryMLFRA, args=(toSendCopy, toSendInd,))
                theThread.start()

            #Reset the pointer once at the end to the start
            if(curInd >= blockSize): curInd = 0
    except Exception as err:
        print(err) 


