#!/usr/bin/env/ python3

#Stores helper functions for the main program. 

from scipy import *
import numpy
import math
import random
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Pose, Twist   

def dist(pose1, pose2): 
    xy_displacement = numpy.array(pose1) - numpy.array(pose2)
    distance_displacement = numpy.hypot(xy_displacement[0], xy_displacement[1])
    return distance_displacement 

#Quat has x,y,z,w fields. Code is adapted from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def quatToEuler(quat):
    #Roll
    sinr_cosp = 2.0 * (quat.w * quat.x + quat.y * quat.z)
    cosr_cosp = 1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    #Pitch
    sinp = 2.0 * (quat.w * quat.y - quat.z * quat.x)
    if ( math.fabs(sinp) >= 1):
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    #Yaw
    siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

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

