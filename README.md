# Robo Final Project
This project navigates the Turtlebot3 through a traffic maze using ROS. It uses the rpi camera onboard to detect the traffic signs using machine learning to determine the correct waypoint for navigation.



Commands to run on the robot (the master):
- run the camera bringup: ```roslaunch turtlebot3_bringup turtlebot3_camera_robot.launch```
- run the navigation stack: ```roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=/home/burger/catkin_ws/src/gryffindor_final_demo/map.yaml open_rviz:=false```
- run the projects launch file in the project directory: ```roslaunch final_demo.launch my_args:=0```


Commands to run on the PC (the slave):
- run rviz for the navigation visualization: ```rosrun rviz rviz -d "/opt/ros/noetic/share/turtlebot3_navigation/rviz/turtlebot3_navigation.rviz"```
