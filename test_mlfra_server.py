import pandas as pd
import time
import socket

debug = True
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

#60 seconds of 100 hz data
blockSize = 60 * 100
filePath = 'data/p02_prox_art.csv'

theMLFRASignal = []
pdDf = pd.read_csv(filePath)

print("Running: " + filePath)

theBlock = []
for index, row in pdDf.iterrows():
    if(len(theBlock) >= blockSize):
        #Start timing
        start = time.time()

        #Convert the array to byte array
        theArr = []
        for x in theBlock: theArr.append(str(x))
        theArr = bytearray((','.join(theArr)).encode('utf-8'))
        
        #Send byte array over socket
        runRes = [-2,-2]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(theArr)

            #Recieve the results
            data = b''
            while True:
                part = s.recv(1024)
                data += part
                if len(part) < 1024: break

            #Decode the results
            runRes = data.decode("utf-8").split(",")
        
        try:
            YN = runRes[0]
            Prob = runRes[1]
        except:
            print(runRes)

        #Calculate elapsed time
        timeElapsed = (round(time.time() - start, 2))

        #Append the signal block
        theMLFRASignal.append([row[1], YN, Prob, timeElapsed])

        print(theMLFRASignal[-1])

        #Clear the block
        theBlock.clear()

    theBlock.append(float(row['value']))

if(debug):
    fout = open("output.csv", 'w')
    fout.write(",".join(["timestamp,YN,Probability,ElapsedTime"]) + "\n")
    for row in theMLFRASignal:
        fout.write(",".join([str(row[0]), str(row[1]), str(row[2]), str(row[3])]) + "\n")
fout.close()
