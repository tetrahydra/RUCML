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

class Timer(object):
    def __init__(self):
        self.time_start = datetime.timedelta(seconds=0)
        self.time_stop = datetime.timedelta(seconds=0)
        self.time_delta = datetime.timedelta(seconds=0)

    def Start(self):
        self.time_start = datetime.datetime.now()

    def Stop(self):
        self.time_stop = datetime.datetime.now()
        self.time_delta = self.time_stop - self.time_start

    def Show(self, text):
        self.Stop()
        print (text + " " + str(time_delta))

    def Value(self):
        self.Stop()
        return self.time_delta