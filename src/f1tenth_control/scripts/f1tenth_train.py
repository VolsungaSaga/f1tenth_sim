#!/usr/bin/env python3

import rospy
import math
import os
import numpy as np
import argparse


from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from std_srvs.srv import Empty



from f1tenth_env import CarEnvironment




if __name__=="__main__":
    rospy.init_node("car_trainer", log_level = rospy.INFO)

    #Parse args
    parser = argparse.ArgumentParser(description="Train a simulated F1Tenth car using DDPG.")
    parser.add_argument('-i', help="Input filename, if you want to train an existing model. Omit the file ending!")
    parser.add_argument('-o', default = "ddpg_f1tenth", help="Output filename for the eventual model. Omit the file ending!")
    parser.add_argument('-check_weight', default= '1.25', type=float, help="Weight for checkpoint reward.")
    parser.add_argument('-gap_weight', default='0.1', type=float, help="Weight for gap reward.")
    parser.add_argument('-safety_weight', default='0.0', type=float, help="Weight for safety reward")
    args = parser.parse_args()

    #Initialize experimental variables
    gamma = 0.99
    tau = 0.001


    #Initialize environment, model.



    env = CarEnvironment(args.check_weight, args.gap_weight, args.safety_weight)
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

    filename = args.i
    if filename is not None and os.path.isfile("./{}.pkl".format(filename)):
        model = DDPG.load(filename, env, gamma=gamma, tau=tau, action_noise=action_noise) #
    else:
        model = DDPG("MlpPolicy", env, gamma=gamma, tau=tau, action_noise=action_noise)
    

    normalize_observations = False
    normalize_returns = False

    #Pause Gazebo here!
    rospy.loginfo("Waiting for Gazebo Pause Physics service...")
    rospy.wait_for_service("/gazebo/pause_physics")
    rospy.loginfo("Gazebo Pause Physics service found!")
    pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    paused = pause_physics()

    #while(not rospy.is_shutdown()):
    rospy.loginfo("Beginning learning process!")
    timesteps = 20000
    save_intervals = 10
    for i in range(save_intervals):
        model.learn(total_timesteps = timesteps / save_intervals)
        model.save(args.o)

    #model.learn(total_timesteps=60000)

    rospy.loginfo("Learning complete, saving to {}.pkl".format(args.o))
    model.save(args.o)