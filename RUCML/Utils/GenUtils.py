import time
import datetime

'''

Usage example:

import RUCML.Utils.Timer as Timer

Timer1 = Timer.Timer()
Timer1.Start()
time_taken = Timer1.Value()
del Timer1 ## Remove object from memory

'''

class PrintLog(object):
    def __init__(self, filename):
        self.filename = filename

    def text(self, text):
        f = open(self.filename,"a+")
        f.write("\n" + str(text))
        f.close()
        print (text)

    def time(self, text):
        f = open(self.filename,"a+")
        f.write("\n" + str(time.strftime("%Y-%m-%d %H:%M:%S")))
        f.close()
        print (text)