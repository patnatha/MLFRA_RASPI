import socket
from mlfra_wrapper import *

#Set the socket parameters
HOST = '127.0.0.1'
PORT = 65432
BUFF_SIZE = 4096

#Open a listening sockey
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    #Loop on connection
    while True:
        dataStream = ""
        conn, addr = s.accept()
        with conn:
            while True:
                #Loop on recieved data to buffer
                data = b''
                while True:
                    part = conn.recv(BUFF_SIZE)
                    data += part
                    
                    #Stop reading if buffer is empty
                    if len(part) < BUFF_SIZE:
                        break

                #Stop processing if buffer was empty
                if(len(data) == 0): break

                #Decode the data and make sure fully recieved
                dataStream += data.decode('utf-8')
                if(len(dataStream.split(',')) == 6000):
                    inputArr = []
                    for x in dataStream.split(','):
                        inputArr.append(float(x))

                    #Run the MLFRA algorithm 
                    print("Received:", len(inputArr), "and running MLFRA!")
                    YN, Prob = MLFRA_Wrapper(inputArr, 100)
                    del inputArr

                    #Returning results
                    conn.sendall(bytearray(','.join([str(YN), str(Prob)]).encode('utf-8')))
                else:
                    print("Partial Recieved:", len(dataStream.split(',')))

