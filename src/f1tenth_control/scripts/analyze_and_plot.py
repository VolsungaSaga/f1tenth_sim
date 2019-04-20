#!/usr/bin/env python3
import rospy
import rosbag
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

def readBagData(filename, limit, ):
    bag = rosbag.Bag(filename)

    if( limit is not None and bag.size() > limit):
        print("This file's size exceeds limit of {} bytes".format(limit))
        return
    count = 0
    msgs = bag.read_messages(["/log/car_pose"])
    num_msgs = bag.get_message_count(["/log/car_pose"])

    x_array = np.array([], dtype=float)
    y_array = np.array([], dtype=float)
    #reward_array = np.array([], dtype=float)
    #time_array = np.array([], dtype=float)
    for msg in msgs:
        print("{}/{}".format(count, num_msgs))
        if msg.topic == "/log/car_pose":
           x_array = np.append(x_array, msg.message.position.x)
           y_array = np.append(y_array, msg.message.position.y)
           count += 1
    bag.close()
    return x_array, y_array
def loadCSV(filename):
    x,y = np.loadtxt(filename, delimiter=",", skiprows=1, unpack=True)
    return x,y
def saveCSV(outfile, data):
    np.savetxt(outfile,data, delimiter=",", header="X,Y")

def main():
    #matplotlib.use("pdf")

    parser = argparse.ArgumentParser(description="Generate simple trajectory plots for a rosbag file using pyplot.")
    parser.add_argument('file_path', help="File with the trajectory data. Must be a .bag file!")
    parser.add_argument('--csv_in',help="You can also specify a csv file, if you want!")
    parser.add_argument('-o', help="Output filename")
    parser.add_argument('--csv_out', help="Save the bag data to a plaintext csv file.")
    parser.add_argument('--size_limit', help="Ignore bag files with size (bytes) greater than this argument's value.", type=int)

    args = parser.parse_args()
    #Args
    filename = args.file_path
    out = args.o
    limit = args.size_limit
    csv_in = args.csv_in
    csv_out = args.csv_out
    
    #Create plot.
    fig, pose_axis = plt.subplots(1,1, squeeze=True)
    #Start reading data from the bag file.
    x_array = np.array([])
    y_array = np.array([])
    if csv_in is not None:
        x_array, y_array = loadCSV(csv_in)
    else:
        x_array, y_array = readBagData(filename, limit)
    
    pose_axis.plot(x_array, y_array, '--r')

    #reward_axis.plot(msg.timestamp.to_sec(), msg.message.data, '.g')

    # Set axis limits to have a square aspect ratio.
    (x_min, x_max) = pose_axis.get_xlim()
    (y_min, y_max) = pose_axis.get_ylim()

    max_limit = max(x_max, y_max)
    min_limit = min(x_min, y_min)
    pose_axis.set_xlim((min_limit,max_limit))
    pose_axis.set_ylim((min_limit,max_limit))

    # Label appropriately
    pose_axis.set_title("Trajectory of Car")
    pose_axis.set_xlabel("X Position (m)")
    pose_axis.set_ylabel("Y Position (m)")
    #reward_axis.set_title("Reward Curve")
    #reward_axis.set_xlabel("Time (s)")
    #reward_axis.set_ylabel("Reward")

    # Boundaries of arena - draw polygons?

    # Export plot to png, save to disk.
    if out is not None:
        plt.savefig(out)

    else:
        plt.savefig("results.png")

    if csv_out is not None:
        saveCSV(csv_out, np.column_stack((x_array,y_array)) )


if __name__ == "__main__":
    main()