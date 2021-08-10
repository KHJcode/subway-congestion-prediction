import sys, os

def block():
    sys.stdout = open(os.devnull, 'w')

def enable():
    sys.stdout = sys.__stdout__
