import socket
import threading
import marshal
import numpy as np
from struct import *

print_ = 0

class MySocket:
    def __init__(self):
        self.HOST = "127.0.0.1"
        self.PORT = 8080
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOST, self.PORT))
        self.s.setblocking(True)


    def sendingMsg(self, action):
        data = pack('f', action)

        #print(data)

        self.s.send(data)


    def getStep(self):
       data = self.s.recv(181*1+3)#184)
       #print(data)

       # 181개의 센서데이터 + vecLen + headingDiff + isDone까지 184개의 데이터가 들어온다
       #pktFormat = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'
       pktFormat = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
       pktSize = calcsize(pktFormat)

       #print("bef : ", len(data))
       #print("pktSize : ", pktSize)

       data_unpack = []
       data_unpack = unpack(pktFormat, data[:pktSize])
        
       #print(data_unpack)

       data_unpack = np.array(data_unpack)

       #states = np.zeros(181+180) 
       states = np.zeros(36+5*2) 
#       states = data_unpack[:-3]    #181센서만
       #reward = data_unpack[-1]
       vecLen = data_unpack[-3]
       headingDiff = data_unpack[-2]
       isDone = data_unpack[-1]

       for i in range(0, 36):
           states[i] = data_unpack[i*5]
       for i in range(36, 36 + 5 ):
           states[i] = vecLen
       for i in range(36 + 5, 36 + 5*2 ):
           states[i] = headingDiff

       if(print_):
           print("vecLen", vecLen)
           print("isDone", isDone)
           for i in range(0, 36+5*2):
               print("[",i ,"] : ", states[i])

#       for i in range(0, 182):
#           states[i] = data_unpack[i]
#           print("[",i ,"] : ", states[i])
#       for i in range(182, 182 + 90 ):
#           states[i] = vecLen
#           print("[",i ,"] : ", states[i])
#       for i in range(182 + 90, 181 + 90*2 ):
#           states[i] = headingDiff
#           print("[",i ,"] : ", states[i])
       #for i in data_unpack:
       #    print("data [", i , "]", data_unpack[i])

       #print("vecLen", vecLen)
       #print("vecLen", data_unpack[181])
       #print("headingDiff", headingDiff)
       #print("headingDiff", data_unpack[182])
       #print("isDone", isDone)
       #print("isDone", data_unpack[183])
       #print("states len : ",len(states))
       #print("  ")
       
       #if(isDone):
           #print(data)
           #print("---------------------------------")
           #print("isDone", isDone)
           #print("---------------------------------")

       data = 0

       #print("-- reward : ", reward)

       return states, vecLen, headingDiff, isDone
