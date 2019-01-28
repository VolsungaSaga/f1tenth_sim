#!/usr/bin/env python3

import rospy
import math

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG

from std_srvs.srv import Empty



from f1tenth_env import CarEnvironment




if __name__=="__main__":
    rospy.init_node("car-master-node", log_level = rospy.INFO)

    #Initialize environment, model.

    env = CarEnvironment()
 
    model = DDPG("MlpPolicy", env)

    #Initialize experimental variables
    gamma = 0.99
    tau = 0.001

    normalize_observations = False
    normalize_returns = False

    #Pause Gazebo here!
    rospy.loginfo("Waiting for Gazebo Pause Physics service...")
    rospy.wait_for_service("/gazebo/pause_physics")
    rospy.loginfo("Gazebo Pause Physics service found!")
    pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    paused = pause_physics()

    while(not rospy.is_shutdown()):

        model.learn(total_timesteps=1000)

        model.save("ddpg-f1tenth")
        rospy.loginfo("Learning complete, saving to ddpg-f1tenth.pkl")