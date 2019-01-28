#!/usr/bin/env python
'''
Author: Robert Pierce

This class, when initialized, will handle the interface between 

'''
import rospy
import geometry_msgs
import std_msgs
import math
import random

from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import LinkStates
#from rosgraph_msgs import Clock

from pybrain.rl.environments.task import Task
from scipy import pi, array, cos, sin



class CarTask():
    
    def linkStatesCallback(self, data):
        twist_x = data.twist[self.chassisLinkIndex].linear.x
        twist_y = data.twist[self.chassisLinkIndex].linear.y
        #We're interested in the magnitude of the velocity, here.
        self.currentSpeed = math.hypot(twist_x, twist_y)

        self.currPos[0] = data.pose[self.chassisLinkIndex].position.x
        self.currPos[1] = data.pose[self.chassisLinkIndex].position.y


    def lidarCallback(self,data):
        self.ranges = data.ranges

    def __init__(self, environment):
        self.chassisLinkIndex = 1 #In the LinkStates message broadcast by Gazebo, the chassis's information is in the 2nd position in the LinkStates arrays.

        self.alpha = 0.5 #The constant that will weigh different facets of the reward - the displacment per unit time portion and the minimum distance from an obstacle portion.


        self.currentSpeed = 0
        self.currPos = [0]*2
        self.prevPos = [0]*2
        self.ranges = []
        #self.timekeeper = 0 #Records how much time has passed between getReward calls. 
        self.linkStatesSub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.linkStatesCallback)
        self.lidarSub = rospy.Subscriber("/scan", LaserScan, self.lidarCallback)
        #self.timekeeperUpdate = rospy.Subscriber("/clock", Clock, self.timekeeperUpdate)

    def getReward(self):
        rospy.loginfo("Timestep complete, rewarding agent.")

        #If currPosition hasn't been updated yet, we still need to return something.
        if(not self.currPos or not self.prevPos or not self.ranges):
            return 0

        #Reward is equal to the total displacement in a given timestep plus the minimum distance from an obstacle
        reward = self.alpha * math.hypot((self.currPos[0] - self.prevPos[0]), (self.currPos[1] - self.prevPos[1])) + (1-self.alpha)*min(self.ranges)
        self.prevPos[0] = self.currPos[0]
        self.prevPos[1] = self.currPos[1]
        #self.timekeeper = 0
        return reward

    #def isFinished(self):
        
