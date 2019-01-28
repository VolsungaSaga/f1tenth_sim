#!/usr/bin/env python
'''
    #Author: Robert Pierce (rtp4qg@virginia.edu) 

This script is in charge of managing the experimentation done when teaching a learner agent how to control a simulated F1/10th race car and navigate a variably shaped track.

The learner agent will be an actor-critic model using OpenAI's implementation 

'''

#ROS Imports
import rospy
#import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

#PyBrain imports
from scipy import *
import sys, time
from pybrain.rl.experiments.continuous import ContinuousExperiment
from pybrain.rl.environments.task import Task


class CarExperiment(ContinuousExperiment):
    def __init__(self, task, agent):
        ContinuousExperiment.__init__(self, task, agent)
        #rospy.init_node('car_experiment')
        self.stepLength = 0.5

        self.currPos = []

        self.prevPos = []
        

    def _oneInteraction(self):
        self.stepid += 1
        self.agent.integrateObservation(self.task.getObservation())
        self.task.performAction(self.agent.getAction())
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty) #Unpause Gazebo here!
        while(not unpause):
            rospy.spinOnce()
        rospy.sleep(self.stepLength) #Sleep for the step length, letting the action we sent garner a result.
        pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty) #Pause Gazebo here!
        while(not pause):
            rospy.spinOnce()

        reward = self.task.getReward()
        self.agent.giveReward(reward)






    
