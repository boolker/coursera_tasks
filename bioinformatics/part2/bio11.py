from __future__ import print_function
import sys, os, math, random
import copy
__author__ = 'drummer'

import matplotlib.pyplot as plt



def to_int(s):
    try:
        return int(s)
    except ValueError:
        return -1

       
def task111():

    input_file_name = os.getcwd() + "/part2/data/10/input21.txt"

    with open (input_file_name, "r") as myfile:
        data=myfile.readlines()

    params = data[0].replace('\n','').split(' ')

    masses = {'A':71,'C':103,'D':115,'E':129,'F':147,'G':57,'H':137,'I':113,'K':128,'L':113,'M':131,'N':114,'P':97,'Q':128,'R':156,'S':87,'T':101,'V':99,'W':186,'Y':163}

    

if __name__ == "__main__":   
    task111() 
    
