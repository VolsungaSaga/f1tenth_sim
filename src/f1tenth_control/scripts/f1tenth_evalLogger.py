#!/usr/bin/env python

'''
This node gathers the data from the log topics published in f1tenth_env.py and 
performs some statistical analysis on it. 

Stats Gathered:

Universal Mean of Min Observation: Average over all time of the observation reported by the 
car.

Rolling Mean: As above, but for a limited interval.

Minimum of Min Observations: Worst case of above.

Average/worst time until goal reached "pylon"

Number of wipeouts over all time.

Number of wipeouts in an interval. 
'''
import rospy
import numpy as np
import math
from f1tenth_util import *
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Pose, Twist

from collections import deque

'''
Parameters:
rolling_interval
goal_model_name
'''
class evalLogger():
    def __init__(self, rolling_interval = 50, goal_model_name = "pylon"):
        #Parameters
        self.rolling_interval = rolling_interval
        self.goal_model_name = goal_model_name
        #Stats
        self.uni_sum_obs = 0. #These are sums of min(observation)
        self.roll_list_obs = deque()

        self.total_num_obs = 0 #Used for calculating avg observations
        self.roll_num_obs = 0 #Incremented until we reach 

        self.worst_obs = np.inf

        self.sum_time_to_goal = 0.  
        
        self.worst_time_to_goal = 0.
        self.num_goals_reached = 0.

        self.time_last_goal = rospy.get_rostime()
        self.duration_last_goal = rospy.Duration(0,0)

        self.uni_num_wipeouts = 0.
        self.roll_mean_wipeouts = 0.


        #Publishers / Subscribers
        self.linkStateSub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.modelStatesCB)
        self.obsSub = rospy.Subscriber("/log/car_obs", Float64MultiArray, self.obsCB)

        self.avg_uni_minObs_pub = rospy.Publisher("/eval/minObsAvg", Float64, queue_size=10)
        self.avg_roll_minObs_pub = rospy.Publisher("/eval/minObsRollAvg", Float64, queue_size=10)
        self.worst_minObs_pub = rospy.Publisher("/eval/minObsWorst", Float64, queue_size=10)

        self.avg_time_toGoal_pub = rospy.Publisher("/eval/timeToGoalAvg", Float64, queue_size=10)
        self.worst_time_toGoal_pub = rospy.Publisher("/eval/timeToGoalWorst", Float64, queue_size=10)

        self.avg_num_wipeouts_pub = rospy.Publisher("/eval/numWipeouts", Float64, queue_size=10)
        self.roll_num_wipeouts_pub = rospy.Publisher("/eval/numeWipeoutsRoll", Float64, queue_size=10)

    def obsCB(self,data):
        #Update observation numbers
        self.total_num_obs += 1
        self.roll_num_obs = min(self.rolling_interval, self.roll_num_obs + 1)
        #Update sums/ rolling list.
        self.uni_sum_obs += min(data.data)
        
        self.roll_list_obs.append(min(data.data))
        if len(self.roll_list_obs) > self.rolling_interval:
            self.roll_list_obs.popleft()

        #Calculate averages
        self.avg_uni_minObs_pub.publish(self.uni_sum_obs / self.total_num_obs)

        self.avg_roll_minObs_pub.publish(sum(self.roll_list_obs) / len(self.roll_list_obs))
        #Output worst min obs.
        if min(data.data) < self.worst_obs:
            self.worst_obs = min(data.data)
        self.worst_minObs_pub.publish(self.worst_obs)

    def modelStatesCB(self,data):
        #Get the car model names
        car_model_name = "f1tenth"
        goal_model_name = self.goal_model_name

        car_pose = data.pose[data.name.index(car_model_name)]
        goal_pose = data.pose[data.name.index(goal_model_name)]

        __,__,car_roll = quatToEuler(car_pose.orientation)

        #Wipeouts
        if car_roll < -1.57 or car_roll > 1.57:
            self.uni_num_wipeouts += 1
        self.avg_num_wipeouts_pub.publish(self.uni_num_wipeouts)
            #TODO: Num wipeouts per some time window.

        #Time to Goal
        if dist([car_pose.position.x, car_pose.position.y], [goal_pose.position.x, goal_pose.position.y]) < 0.2:
            
            time_now = rospy.get_rostime()
            #Gazebo doesn't reset the sim time when /reset_world service is called, so we need
            # to do some timekeeping.
            duration_this_goal = time_now - self.time_last_goal
            #Update worst time to goal if necessary.
            if duration_this_goal > duration_last_goal:
                self.worst_time_to_goal = duration_this_goal.to_sec()
            self.worst_time_toGoal_pub.publish(self.worst_time_to_goal)
            #Compute, publish average.
            self.num_goals_reached += 1
            self.sum_time_to_goal += duration_this_goal.to_sec()
            avg_time_goal = self.sum_time_to_goal / self.num_goals_reached
            self.avg_time_toGoal_pub.publish(avg_time_goal)

            #Update persistent variables
            self.duration_last_goal = duration_this_goal
            self.time_last_goal += duration_this_goal

    



def main():
    rospy.init_node("eval_logger", log_level=rospy.DEBUG)
    logger = evalLogger()
    loop_rate = rospy.Rate(125)
    while not rospy.is_shutdown():
        loop_rate.sleep()
    rospy.spin()

if __name__ == "__main__":
    main()
    