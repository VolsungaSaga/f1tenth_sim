#!/usr/bin/env python
'''
	#Author: Robert Pierce (rtp4qg@virginia.edu) 

This script is in charge of managing the experimentation done when teaching a learner agent how to control a simulated F1/10th race car and navigate a variably shaped track.

The learner agent will be a Neural Network trained using Q-Learning, using the PyBrain reinforcement learning libraries. 

'''

#ROS Imports
import rospy
import math
import random
#import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan


#PyBrain imports
from scipy import *
import sys, time

#We are using a Reinforcement Learner
from pybrain.rl.learners.valuebased import ActionValueNetwork
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.environments.episodic import EpisodicTask
from scipy import pi, array, cos, sin


class f1tenth_experiment(EpisodicExperiment):


	
