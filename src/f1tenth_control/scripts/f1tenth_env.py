#!/usr/bin/env/ python3
## Environment that will interface with OpenAI. As all physics, rendering, and control are handled by Gazebo7, 
## this environment will only query and command the Gazebo application. 

#ROS Imports
import rospy
import math
import random
import angles
#import matplotlib.pyplot as plt
from f1tenth_util import *
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Pose, Twist

#from tf.transformations import euler_from_quaternion

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

    #Quat has x,y,z,w fields. Code is adapted from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    def quatToEuler(self,quat):
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

    def getObs(self):
        if self.currObs is not None:
            return self.currObs

    def modelStatesCallback(self, data):
        #First, we'll want to figure out where the car chassis is in the array.
        model_name = "f1tenth"
        model_index = data.name.index(model_name)


        twist_x = data.twist[model_index].linear.x
        twist_y = data.twist[model_index].linear.y
        #We're interested in the magnitude of the velocity, here.
        self.currentSpeed = math.hypot(twist_x, twist_y)

        self.twist = data.twist[model_index]

        self.currPos[0] = data.pose[model_index].position.x
        self.currPos[1] = data.pose[model_index].position.y

        self.roll, self.pitch, self.yaw = self.quatToEuler(data.pose[model_index].orientation)
        rospy.logdebug("Car Yaw:{}".format(self.yaw))



        #Initialize/update checkpoint stuff.
        pylon_name_gz = "pylon"
        if len(self.checkpointList) == 0:
            index = data.name.index(pylon_name_gz)
            self.checkpointList.append([data.pose[index].position.x, data.pose[index].position.y])
            self.currCheckpoint = self.checkpointList[0]
            
            if len(self.checkpointList) <= 0:
                rospy.logerr("No checkpoints found in this environment!")
                rospy.shutdown() 

        #We also want to calculate the relative heading of the checkpoint.
        car_goal_xdiff = self.currPos[0] - self.currCheckpoint[0]
        car_goal_ydiff = self.currPos[1] - self.currCheckpoint[1]

        #TODO : When you come back from Georgia, reverse the vector!
        car_facing_vec = [-1 * math.cos(self.yaw), -1 * math.sin(self.yaw)]

        theta_c_g = math.atan2(car_facing_vec[1], car_facing_vec[0]) - math.atan2(car_goal_ydiff, car_goal_xdiff)
        self.relative_yaw = theta_c_g
        rospy.logdebug("Rel Angle Car_Goal: {}".format(theta_c_g))


        #Publish car pose info to log topic.
        self.logPosition.publish(data.pose[model_index])

        self.logVelocity.publish(data.twist[model_index])

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
        numpy.append(self.currObs,self.relative_yaw)
        #rospy.loginfo(len(self.currObs))
        obsMsg = Float64MultiArray()
        obsMsg.data = self.currObs

        self.logObservation.publish(obsMsg)

    def getArcsFromObs(self, obs):
        #Return everything except last one, which is the rel angle heading.
        return obs[0:-1]


    #We also pass in a dictionary of weights
    def __init__(self, checkWeight = 1.25, gapWeight = 0.1, safetyWeight = 0.0 ):
        #rospy.init_node("car-environment")
        self.currentLIDARReading = [1]*1081
        self.currObs = None

     	self.time_step_size = rospy.get_param("/time_step_size")
        self.update_rate = rospy.get_param("/update_rate")

        self.dt_real = 0.2 / (self.time_step_size * self.update_rate)

        #Some constants to make our lives a bit easier
        self.cellsPerDegree = 6 #How many LIDAR cells does it take to cover one degree?
        self.degreesPerArc = 1 #The size of the arcs in degrees.

        self.minExpectedProgress = 0.01 #How much displacement do we expect per time step, in meters, at minimum? If the car doesn't go at least this distance, the episode is considered over.
        self.minDistFromObstacle = 0.2

        #Action, State Space defs:
        self.maxSpeed = 1.0
        self.minSpeed = 0.0
        self.maxSpeedNorm = 1.0
        self.minSpeedNorm = -1.0
        self.maxAngle = 90.0
        self.minAngle = -90.0

        self.maxSense = numpy.inf #We don't consider there to be a bound on sensor readings.
        self.minSense = 0.0

        # The action space consists of two variables: Vehicle Speed and Steering Angle
        self.action_space = spaces.Box(low=numpy.array([self.minSpeedNorm, self.minAngle]), high=numpy.array([self.maxSpeedNorm,self.maxAngle]), dtype=float)
        # The observation space consists of each arc that we divided the LIDAR readings into.
        self.observation_space = spaces.Box(low=self.minSense, high=self.maxSense, shape=([math.floor(1081/(self.cellsPerDegree * self.degreesPerArc)) + 1]), dtype=float)

        ###REWARD FUNCTION CONSTANTS / VARIABLES

        self.alpha = 0.2 #The tuning parameter that will weigh different facets of the reward.

        self.checkpointWeight = checkWeight
        self.gapWeight = gapWeight
        self.safetyWeight = safetyWeight

        self.currCheckpoint = None
        self.checkpointList = []
        self.checkpointDistTolerance = 0.1
        self.checkpointLoop = False

        self.move_timer = rospy.Duration(0.0)
        self.move_timer_started = False

        #Actions, Position, Pose
        self.currentAction_Speed = 0
        self.currentAction_Steer = 0
        self.currPos = [0]*2
        self.prevPos = [-1]*2

        self.roll = 0.
        self.pitch = 0.
        self.yaw = 0. #Roll, pitch, yaw

        self.relative_yaw = 0.

        self.currentSpeed = 0.
        self.twist = Twist()
        #self.timekeeper = 0 #Records how much time has passed between getReward calls. 
        self.scanSub = rospy.Subscriber("scan",LaserScan, self.scanCallback)

        self.ackermannPub = rospy.Publisher("~/ackermann_cmd", AckermannDriveStamped, queue_size=5)

        self.debugExpectedDisplacement = rospy.Publisher("/debug/expDisplacement", Float64, queue_size=5)
        self.debugActualDisplacement = rospy.Publisher("/debug/actDisplacement", Float64, queue_size=5)
        self.logReward = rospy.Publisher("/log/reward", Float64, queue_size=5)
        self.logPosition = rospy.Publisher("/log/car_pose", Pose, queue_size = 5)
        self.logVelocity = rospy.Publisher("/log/car_velocity", Twist, queue_size = 5)
        self.logObservation = rospy.Publisher("/log/car_obs", Float64MultiArray, queue_size = 10)

        self.linkStatesSub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.modelStatesCallback)

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
        xy_displacement = numpy.array(currPos) - numpy.array(prevPos)
        distance_displacement = numpy.hypot(xy_displacement[0], xy_displacement[1])
        return distance_displacement        


    def getReward(self):
        rospy.logdebug("Timestep complete, rewarding agent.")

        reward = self.checkpointReward(self.checkpointWeight, self.currCheckpoint) + self.gapReward(self.gapWeight, self.getArcsFromObs(self.currObs)) + self.safetyReward(self.safetyWeight, self.getArcsFromObs(self.currObs))

        self.logReward.publish(Float64(reward))
        return reward

    #Safety Reward: Basically, reward for staying away from the walls and staying upright.
    def safetyReward(self, alpha, observation):
        if(self.roll < -1.57 or self.roll > 1.57):
            return alpha * -10000

        return alpha * 10*math.log(self.clamp(min(observation) - self.minDistFromObstacle, 0.01, 100))
    #Displacement Reward: Reward is based on the car's immediate displacement
    # in the xy plane, along with . 
    def displacementReward(self, alpha, currPos, prevPos):
        reward = alpha *self.dist(currPos, prevPos)
        if(self.dist(currPos, prevPos) < self.minExpectedProgress):
            reward -= 10000
        return reward

    #Gap Reward: A Reward for keeping the front of the car clear of obstacles.
    def gapReward(self, alpha, observation):
        leftIndex = len(observation) // 3
        rightIndex = (len(observation) // 3) * 2
        middleObs = observation[leftIndex:rightIndex]

        reward = alpha * 10*math.log(self.clamp(min(middleObs), 0.0001, 100))
        return reward

    # Checkpoint: [x , y]
    def checkpointReward(self, alpha, currCheckpoint):
        #reward = alpha * -self.dist(self.currPos, currCheckpoint) + 50

        #Get vector of car motion
        car_velocity_vector = self.twist.linear

        #Get vector of goal position relative to car.
        car_goal_diff = [currCheckpoint[0] - self.currPos[0], currCheckpoint[1] - self.currPos[1]]  
        car_to_goal_vector = Vector3(car_goal_diff[0], car_goal_diff[1], 0.)

        #Normalize both.
        norm_vel_vector = normalize_vector(car_velocity_vector)
        norm_goal_vector = normalize_vector(car_to_goal_vector)
        #Do dot product
        dot = dotProduct(norm_vel_vector, norm_goal_vector)
        rospy.logdebug("**Dot Product {}**".format(dot))

        reward = alpha * dot

        #Return scaled reward
        return reward

    def getDone(self):
        #Check for uninitialized currPos, prevPos
        if(not self.currPos or not self.prevPos):
            return False

        rospy.logdebug("Current Speed (Abs):" + str(self.currentSpeed))
        
        #If we don't want to loop, we're done after we reach final checkpoint.
        if(self.currCheckpoint is not None and self.checkpointLoop is False and self.checkpointList.index(self.currCheckpoint) == len(self.checkpointList) -1):
            if(self.dist(self.currPos, self.currCheckpoint) < self.checkpointDistTolerance):
                rospy.loginfo("Car: "+ str(self.currPos))
                rospy.loginfo("Goal:" + str(self.currCheckpoint))
                rospy.loginfo("Distance to Goal:" + str(self.dist(self.currPos, self.currCheckpoint)))
                rospy.loginfo("We've reached the goal. Trial complete.")
                return True

        #New plan! We only stop if we've wiped out - which happens pretty easily.
        if(self.roll < -1.57 or self.roll > 1.57):
            rospy.logdebug("We've wiped out. Trial complete.")
            return True

        #If we make no progress for 10 seconds, reset!
        if(self.dist(self.currPos, self.prevPos) < 0.01): 
            if self.move_timer_started == False:
                self.move_timer_started = True
                self.move_timer = rospy.Time.now()
            else:
                if( rospy.Time.now() - self.move_timer > rospy.Duration(10.0)):

                    rospy.loginfo("Car hasn't moved for 10 seconds! Resetting!")
                    self.move_timer_started = False
                    return True

        else:
            self.move_timer_started = False

        return False




    #For resetting the environment, we just need to send the appropriate service call to Gazebo ROS.

    def reset(self):
        reset = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        resetted = reset()
        while(not reset):
            rospy.spinOnce()

 

        #Reinit current positon, previous position
        self.currPos = [0,0]
        self.prevPos = [-1,-1]

        if self.currObs is not None:
            return self.getObs()
        else:
            rospy.logwarn("In env.reset(): self.currObs is None! Waiting for observations... ")
            time_start = rospy.get_rostime()
            unpause_physics = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
            unpaused = unpause_physics()

            while(self.currObs is None or len(self.currObs) != math.floor(1081/(self.cellsPerDegree * self.degreesPerArc)) + 1):
                rospy.sleep(1)

            rospy.logwarn("Observations found {}! Resuming...".format(len(self.currObs)))
            return self.getObs()

    def render(self):
        #Do nothing, since we already have Gazebo to render the environment.
        pass

    def step(self,max_action):
        self.performAction(max_action)

        unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty) #Unpause Gazebo here!
        unpaused = unpause_physics()
        while(not unpaused):
            rospy.spinOnce()
        rospy.sleep(self.dt_real) #Sleep for the step length, letting the action we sent garner a result.
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

