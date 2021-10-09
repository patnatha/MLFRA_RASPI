import os
import sys
from subprocess import Popen, DEVNULL, PIPE, run

def stopMLClient():
    cmd_str = "ps -aux | grep client_32 | awk '{print $2}'"
    proc = run([cmd_str], shell=True, stderr=DEVNULL, stdout=PIPE)

    try:
        #Extract the PID to kill
        theVals = proc.stdout.decode('utf-8').strip("\n")
        firstVal = theVals.strip("\n").split("\n")

        for thePID in firstVal:
            kproc = run(["sudo kill " + str(thePID)], \
                         shell=True, stderr=PIPE, stdout=PIPE)
    except:
        print("Client not running, couldn't find PID")
   
stopMLClient() 
