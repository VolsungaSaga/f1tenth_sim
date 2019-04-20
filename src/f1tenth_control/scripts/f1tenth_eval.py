#!/usr/bin/env python3

import rospy
import math
import os
import argparse

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from std_srvs.srv import Empty
from f1tenth_env import CarEnvironment

if __name__=="__main__":
    rospy.init_node("car_evaluator", log_level = rospy.DEBUG)
    

    #Parse args
    parser = argparse.ArgumentParser(description="Evaluate a simulated F1Tenth car's trained DDPG model.")
    parser.add_argument('file', help="Input filename for the model to evaluate. Omit the file ending!")
    
    args = parser.parse_args()

    #Initialize experimental variables
    gamma = 0.99
    tau = 0.001


    #Initialize environment, model.

    env = CarEnvironment()
    filename = args.file
    if os.path.isfile("{}.pkl".format(filename)):
        model = DDPG.load(filename, env, gamma=gamma, tau=tau) #
    else:
        raise IOError("Could not find {}.pkl".format(filename))

    normalize_observations = False
    normalize_returns = False

    #Pause Gazebo here!
    rospy.loginfo("Waiting for Gazebo Pause Physics service...")
    rospy.wait_for_service("/gazebo/pause_physics")
    rospy.loginfo("Gazebo Pause Physics service found!")
    pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    paused = pause_physics()

    obs = env.reset()
    while(not rospy.is_shutdown()):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if any(dones):
            rospy.wait_for_service("/gazebo/reset_world")
            rospy.loginfo("Resetting world!")
            reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
            resetted = reset_world()
        env.render()

    rospy.loginfo("Evaluation complete!")