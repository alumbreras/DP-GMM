# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:10:08 2015
Collects the status of every worker and prints a summary and save it in file monitor.txt

@author: alumbreras
"""

import os
import time

line = time.strftime("%Y-%m-%d %H:%M:%S") + "\n"
print line,

for fname in sorted(os.listdir(".workers/")):
    if fname.endswith(".progress"):
        with open(".workers/"+fname, "r") as fi:
            progress = fi.readline()
            line = str(fname) + ": "+ str(progress) +"\n"
            print line,
        
