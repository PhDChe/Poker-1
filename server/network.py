"""
 * Sets up local network socket
 * Hopefully works with hamachi so there wont be any need
 * to open up ports
 https://github.com/techwithtim/Network-Game-Tutorial/blob/master/network.py
"""

import socket

class network:
    def __init__(self, hamachiAddr):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = str(hamachiAddr)
        self.port = 24                   #32976
        self.addr = (self.host, self.port)
        self.id = self.connect()
        

    def connect(self):
        self.client.connect(self.addr)
        return self.client.recv(2048).decode()
        

    def send(self, data):
        try:
            self.client.send(str.encode(data))
            reply = self.client.recv(2048).decode()
            return reply

        except socket.error:
            return
        