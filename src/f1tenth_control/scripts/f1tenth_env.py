#!/usr/bin/env/ python3
## Environment that will interface with OpenAI. As all physics, rendering, and control are handled by Gazebo7, 
## this environment will only query and command the Gazebo application. 

#ROS Imports
import rospy
import math
import random
#import matplotlib.pyplot as plt
from f1tenth_util import *
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose, Twist

from gym import core, spaces
from gym import error, utils


#TensorFlow imports
from scipy import *
import numpy
import sys, time

class CarEnvironment():

    def clamp(self,x, minVal, maxVal):
        return max(min(x, maxVal), minVal)
    def normalizeRange(self, x, a, b, x_min, x_max):
        #Given a raw measurement (and raw min/max), normalize it into the range specified by [a,b]
        ret = ((b-a)*(x-x_min))/(x_max - x_min) + a
        return ret

    def isArcOccupied(self,arc):
        return numpy.mean(arc) < 5

    def getArcs(self,n,ranges): 
        #Create an array of arcs -- slices of the original array.
        arcs = numpy.array([data.ranges[i:i+n] for i in range(0, len(data.ranges), n)])
        return arcs

    def getObs(self):
        if self.currObs is not None:
            return self.currObs

    def linkStatesCallback(self, data):
        twist_x = data.twist[self.chassisLinkIndex].linear.x
        twist_y = data.twist[self.chassisLinkIndex].linear.y
        #We're interested in the magnitude of the velocity, here.
        self.currentSpeed = math.hypot(twist_x, twist_y)

        self.currPos[0] = data.pose[self.chassisLinkIndex].position.x
        self.currPos[1] = data.pose[self.chassisLinkIndex].position.y

        #rospy.logdebug("UPDATE: CurrPos = [%.2f, %.2f]", self.currPos[0], self.currPos[1])

    def scanCallback(self, data):
        #Get however many arcs. 
        n = self.degreesPerArc * self.cellsPerDegree
        arcs = numpy.array([data.ranges[i:i+n] for i in range(0, len(data.ranges), n)]) #getArcs(self.cellsPerDegree * self.degreesPerArc, data.ranges)
        #rospy.loginfo(arcs)
        #Craft array consisting of the distance reading averages of each arc.
        arc_avgs = numpy.array([])
        for arc in arcs:
            arc_avgs = numpy.append(arc_avgs,numpy.mean(arc))

        #rospy.loginfo(arc_avgs)


        self.currObs = arc_avgs
        #rospy.loginfo(data.ranges)



    def __init__(self):
        #rospy.init_node("car-environment")
        self.currentLIDARReading = [1]*1081
        self.currObs = None


        self.stepLength = 0.2
        #Some constants to make our lives a bit easier
        self.cellsPerDegree = 6 #How many LIDAR cells does it take to cover one degree?
        self.degreesPerArc = 1 #The size of the arcs in degrees.

        self.minExpectedProgress = 0.01 #How much displacement do we expect per time step, in meters, at minimum? If the car doesn't go at least this distance, the episode is considered over.
        self.minDistFromObstacle = 0.2


        self.chassisLinkIndex = 1 #In the LinkStates message broadcast by Gazebo, the chassis's information is in the 2nd position in the LinkStates arrays.

        #Action, State Space defs:
        self.maxSpeed = 10.0
        self.minSpeed = 0.0
        self.maxSpeedNorm = 1.0
        self.minSpeedNorm = -1.0
        self.maxAngle = 90.0
        self.minAngle = -90.0

        self.maxSense = numpy.inf #We don't consider there to be a bound on sensor readings.
        self.minSense = 0.0



        # The action space consists of two variables: Vehicle Speed and Steering Angle
        self.action_space = spaces.Box(low=numpy.array([self.minSpeedNorm, self.minAngle]), high=numpy.array([self.maxSpeedNorm,self.maxAngle]))
        # The observation space consists of each arc that we divided the LIDAR readings into.
        self.observation_space = spaces.Box(low=self.minSense, high=self.maxSense, shape=([math.ceil(1081/(self.cellsPerDegree * self.degreesPerArc))]))

        ###REWARD FUNCTION CONSTANTS

        self.alpha = 0.2 #The tuning parameter that will weigh different facets of the reward - the displacment per unit time portion and the minimum distance from an obstacle portion.


        #Actions, Position
        self.currentAction_Speed = 0
        self.currentAction_Steer = 0
        self.currPos = [0]*2
        self.prevPos = [-1]*2

        self.currentSpeed = 0
        #self.timekeeper = 0 #Records how much time has passed between getReward calls. 
        self.linkStatesSub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.linkStatesCallback)
        self.scanSub = rospy.Subscriber("scan",LaserScan, self.scanCallback)

        self.ackermannPub = rospy.Publisher("~/ackermann_cmd", AckermannDriveStamped, queue_size=5)

        self.debugExpectedDisplacement = rospy.Publisher("/debug/expDisplacement", Float64, queue_size=5)
        self.debugActualDisplacement = rospy.Publisher("/debug/actDisplacement", Float64, queue_size=5)
        self.logReward = rospy.Publisher("/log/reward", Float64, queue_size=5)
        self.logPosition = rospy.Publisher("/log/car_pose", Pose, queue_size = 5)
        self.logVelocity = rospy.Publisher("/log/car_velocity", Twist, queue_size = 5)
        self.logObservation = rospy.Publisher("/log/car_obs", Float64Array, queue_size = 10)

        rospy.loginfo("Environment initialization complete!")

    #Here, we'll assume that an action is just two floating point values:
    # Action: [Steering Angle, ]
    def performAction(self,action):
        #rospy.loginfo("In performAction(action):")
        #rospy.loginfo("Size of action array:" + str(len(action)))
        rospy.loginfo("Suggested Action: " + str(action))

        drivemsg = AckermannDriveStamped()
        drivemsg.header.stamp = rospy.Time.now()
        drivemsg.header.frame_id = "base_link"
        #The first parameter, speed, is on a -1,1 scale to accomodate DDPG's need to have equal in magnitude minimum and maximum action values
        #Therefore, I will denormalize it before sending it out.
        actual_speed = self.normalizeRange(action[0], self.minSpeed, self.maxSpeed, self.minSpeedNorm, self.maxSpeedNorm)
        rospy.loginfo("Suggested Speed: " + str(actual_speed))
        drivemsg.drive.speed = actual_speed
        #drivemsg.drive.speed = 50

        drivemsg.drive.acceleration = 1
        drivemsg.drive.jerk = 1
        drivemsg.drive.steering_angle = action[1]
        drivemsg.drive.steering_angle_velocity = 1

        self.currentAction_Speed = action[0]
        self.currentAction_Steer = action[1]

        self.ackermannPub.publish(drivemsg)

    def dist(self, currPos, prevPos):
        xy_displacement = numpy.array(self.currPos) - numpy.array(self.prevPos)
        distance_displacement = numpy.hypot(xy_displacement[0], xy_displacement[1])
        return distance_displacement        


    def getReward(self, FUNCTION): #Function is a string that we'll check.
        rospy.loginfo("Timestep complete, rewarding agent.")
        #Reward is equal to the total displacement in a given timestep plus the minimum distance from an obstacle
        reward = self.displacementReward(self.alpha, self.currPos, self.prevPos) + self.safetyReward(1-self.alpha, self.currObs) #(1-self.alpha)* 10*math.log(self.clamp(min(self.currObs) - self.minDistFromObstacle, 0.01, 100))
        if(self.dist(self.currPos, self.prevPos) < self.minExpectedProgress):
            reward -= 10000
        rospy.logdebug("Reward:" + str(reward))
        self.logReward.publish(Float64(reward))
        #self.timekeeper = 0
        return reward

    #Safety Reward: Basically, reward for staying away from the walls.
    def safetyReward(self, alpha, observation):
        return (1-alpha)* 10*math.log(self.clamp(min(observation) - self.minDistFromObstacle, 0.01, 100))
    #Displacement Reward: Reward is based on the car's immediate displacement
    # in the xy plane, along with . 
    def displacementReward(self, alpha, currPos, prevPos):
        reward = alpha *self.dist(currPos, prevPos)
        if(self.dist(currPos, prevPos) < self.minExpectedProgress):
            reward -= 10000
        return reward

    #Gap Reward: A Reward for keeping the front of the car clear of obstacles.
    def gapReward(self, alpha, observation):
        #TODO: Get middle of observation.
        leftIndex = len(observation) // 3
        rightIndex = (len(observation) // 3) * 2
        middleObs = observation[leftIndex:rightIndex]

        reward = alpha * 10*math.log(self.clamp(min(middleObs), 0.01, 100))

    # Checkpoint: [x , y]
    def checkpointReward(self, alpha, checkpoint):
        reward = alpha * self.dist(self.currPos, checkpoint)
        return reward

    def getDone(self):
        # The experiment is done when the car has not made any forward progress in the last time step.
        if(not self.currPos or not self.prevPos):
            return False

        #Case 1: No progress.
        distance_displacement = self.dist(self.currPos, self.prevPos)
        expected_displacement = self.currentSpeed * self.stepLength
        
        rospy.logdebug("Current Speed (Abs):" + str(self.currentSpeed))
        #If we haven't undergone any displacement and our speed is non-zero, we must be running into a wall.
        if(math.fabs(distance_displacement) <= 0.005):
            return True
        else:
            return False




    #For resetting the environment, we just need to send the appropriate service call to Gazebo ROS.

    def reset(self):
        reset = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        resetted = reset()
        while(not reset):
            rospy.spinOnce()

        if self.currObs is not None:
            return self.getObs()
        else:
            rospy.logerr("In env.reset(): self.currObs is None! ")

        #Reinit current positon, previous position
        self.currPos = [0,0]
        self.prevPos = [-1,-1]

    def render(self):
        #Do nothing, since we already have Gazebo to render the environment.
        pass

    def step(self,max_action):
        self.performAction(max_action)

        unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty) #Unpause Gazebo here!
        unpaused = unpause_physics()
        while(not unpaused):
            rospy.spinOnce()
        rospy.sleep(self.stepLength) #Sleep for the step length, letting the action we sent garner a result.
        pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty) #Pause Gazebo here!
        paused = pause_physics()
        while(not paused):
            rospy.spinOnce()

        new_obs = self.getObs()
        reward = self.getReward()
        done = numpy.array([self.getDone()])
        info = "Envs finished:" + str(done)
        #Update previous position.
        self.prevPos = self.currPos[:]


        return new_obs, reward, done, info

