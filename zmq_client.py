import zmq
import numpy as np
from urllib.request import urlopen, Request
from urllib.parse import urlparse

port = "5558"
ip = "192.168.6.21"

arr = np.array([1.02, 1.55, 1.89])

context = zmq.Context()
print ("Connecting to server...")
socket = context.socket(zmq.REQ)
socket.connect ("tcp://%s:%s" % (ip, port))

# p1 = "/shared/saurabh.m/101_ObjectCategories/airplanes/image_0002.jpg"
img = "https://images.wehkamp.nl/i/wehkamp/16443748_eb_03"

string = img.encode("utf_8")
socket.send(string)
msg = socket.recv()

s2_soft = np.frombuffer(msg, dtype = np.float32)
print(s2_soft.shape)