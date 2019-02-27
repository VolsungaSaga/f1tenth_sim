#!/usr/bin/env/ python3

#Stores helper functions for the main program. 

from scipy import *
import numpy
import math
import random

def getCarDisplacement(currPos, prevPos):
    xy_displacement = numpy.array(currPos) - numpy.array(prevPos)
    distance_displacement = numpy.hypot(xy_displacement[0], xy_displacement[1])
    return distance_displacement    

def clamp(x, minVal, maxVal):
    return max(min(x, maxVal), minVal)
def normalizeRange(x, a, b, x_min, x_max):
    #Given a raw measurement (and raw min/max), normalize it into the range specified by [a,b]
    ret = ((b-a)*(x-x_min))/(x_max - x_min) + a
    return ret

def isArcOccupied(arc):
    return numpy.mean(arc) < 5

def getArcs(n,ranges): 
    #Create an array of arcs -- slices of the original array.
    arcs = numpy.array([data.ranges[i:i+n] for i in range(0, len(data.ranges), n)])
    return arcs


def displacementReward( currPos, prevPos, weight):
    return weight*displacement
def safetyReward( distFromWall, minDistThreshold, weight):
    return weight * 10*math.log(clamp(min(distFromWall - minDistThreshold, 0.01, 100)))

def gapReward(observation, weight)