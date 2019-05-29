Using Reinforcement Learning to Drive an Autonomous Car
----------------------------------------------------------

Instructions for Setup/Use

In order to use this package, you will need a couple of packages.

ROS Dependencies:
- ROS Kinetic Kame
- ros-control
- gazebo7
Other Dependencies:
- TensorFlow
- NumPy
- SciPy
- Stable-Baselines (a fork of OpenAI)

ROS Kinetic uses Python2.7, while TensorFlow and Stable-Baselines use Python3. If you attempt to run the training script without using virtual environments to isolate each Python version and their libraries, you won't succeed.

Essentially, you want to create two environments (I recommend using virtualenv for creating them). The system environment (or the 1st virtual environment) should have Python2.7 and ROS Kinetic installed, while the 2nd virtual environment should have Python3 and stable-baselines installed. If you use pip to install stable-baselines, it should install all the dependencies along with it.
