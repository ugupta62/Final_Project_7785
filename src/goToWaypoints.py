#!/usr/bin/env python
import rospy
import numpy as np
import sys
import cv2 as cv
from time import time, sleep
from copy import copy, deepcopy
# from scipy.spatial.transform import Rotation as R

from knn_image_classification import ImageProcess

from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from actionlib_msgs.msg import GoalStatusArray

'''Mahdi Ghanei and Ujjwal Gupta
'''

def wrapToPi(ang):
	"""wraps angle(rad) to [-pi,pi]
	Args:
		ang (float): angle to wrap in rad

	Returns:
		[float]: wrapped angle
	"""
	if ang < -np.pi:
		ang += 2*np.pi
	elif ang > np.pi:
		ang -= 2*np.pi

	return ang

class MazeNavigator:
	def __init__(self):
		"""The init method for the MazeNavigator class
		"""

		### the variables for data callback data storage
		self.image_np = None
		self.lidar_arr = None
		self.robot_pose = None				# current robot pose (type is PoseWithCovarianceStamped)

		### the publishers
		self.pub_nav = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size = 1)
		self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size = 5)

		self.pub_maskimg = rospy.Publisher('/mask_img', CompressedImage, queue_size = 1)

		### the subscribers
		rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callbackAMCLPose,  queue_size = 1)
		rospy.Subscriber('/scan', LaserScan, self.callbackLidar)
		rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.callbackCam, queue_size = 1, buff_size=2**24)
		rospy.Subscriber('/move_base/status', GoalStatusArray, self.callbackNav, queue_size=1)


		### read waypoints
		self.waypoints = rospy.get_param('/waypoints')
		for i in range(len(self.waypoints)): # negative sign for ys
			for j in range(len(self.waypoints[i])):
				self.waypoints[i][j][1] = -self.waypoints[i][j][1]
		print('waypoints read\n', self.waypoints)

		### train the classifier
		self.img_process = ImageProcess()
		self.img_process.trainClassifier()

		### current robot status
		while self.robot_pose is None:
			pass
		print('\n\n\nfinding first waypoint*****************************************')
		self.cur_w_ind = None
		self.set_cur_w_ind(self.findClosestWaypoint())				# current waypoint index
		print('first index', self.get_cur_w_ind())

		### variables for the Nav. stack
		self.nav_checkpoint = 0
		self.nav_previous_checkpoint = -1

		self.nav_status= None
		self.nav_previous_status = None
		self.nav_reject_first_status = "Yes"
		print('init method complete')


	def findClosestWaypoint(self):
		"""Finds the index of the closest waypoint.

		Returns:
			[int]: the index of the closestwaypoint
		"""
		robot_pose = self.get_robot_pose()

		x = robot_pose.pose.pose.position.x
		y = robot_pose.pose.pose.position.y
		a = np.array([x,y])
		
		min_dist = np.inf
		min_ind = (0,0)
		for i in range(len(self.waypoints)):
			for j in range(len(self.waypoints[i])):
				b = np.array([self.waypoints[i][j][0], self.waypoints[i][j][1]])
				dist = np.linalg.norm(a-b)
				if dist < min_dist:
					min_dist = dist
					min_ind = (i,j)
		i = min_ind[0]
		j = min_ind[1]
		print(a, np.array([self.waypoints[i][j][0], self.waypoints[i][j][1]]))
		return min_ind

	def waypointSanityCheck(self):
		'''Checks if the current waypoint index matches the location of the robot.
		'''
		ind = self.findClosestWaypoint()
		assert ind == self.cur_w_ind, 'Current waypoint index mismatch. Closest={}, Curr={}'.format(ind, self.cur_w_ind)
					
	def isWall(self, ang=0):
		"""Checks if there is a wall in the direction sepcified using Lidar and returns minimum dist.
		 Should be called to ensure the sign indeed exists

		Args:
			ang (int, optional): the angle to check for wall. Defaults to 0.

		Returns:
			[bool, float, float]:
		"""
		is_wall = False
		min_dist = np.inf
		min_ang = None
		for i in range(360):
			if self.lidar_arr.ranges[i] <= min_dist and self.lidar_arr.ranges[i] > 0.0:
				min_dist = self.lidar_arr.ranges[i]
				min_ang = i
		
		if (self.lidar_arr.ranges[ang] > 0 and self.lidar_arr.ranges[ang] < 0.85):
			is_wall = True

		min_ang = wrapToPi(min_ang * (2*np.pi/180.0))

		return is_wall, min_dist, min_ang


	def getNextWaypoint(self, traffsign, update_w_ind=True):
		"""Determine the adjacent waypoint based on traffic sign

		Args:
			traffsign (string): the traffic sign
			update_w_ind (bool, optional): whether to update the waypoint index (False for rotation goal). Defaults to True.

		Returns:
			[list, float]: next waypoint and heading
		"""
		ind = self.get_cur_w_ind()
		x_pose,y_pose,heading = self.calcRobotPose()

		# TODO: the signs might be reversed
		if traffsign == 'left':
			heading += np.pi/2.0
		elif traffsign == 'right':
			heading += -np.pi/2.0
		elif traffsign == 'stop' or traffsign == 'dont_enter':
			heading += np.pi
		elif traffsign == 'empty':
			heading += 0
		heading = wrapToPi(heading)

		# handle the contradiction where iswall but no sign detected
		if (not update_w_ind) and traffsign=='empty': # if is_wall
			heading += np.pi/2.0
			print('handling contradiction')
		
		# round heading to a right angle (to correct for deviations)
		round_base = np.pi/2.0
		heading = round_base * round(heading/round_base)


		temp_cur_w_ind = self.get_cur_w_ind()
		y_ind = temp_cur_w_ind[1] + np.cos(heading)
		x_ind = temp_cur_w_ind[0] - np.sin(heading)
		ind = (int(round(x_ind)), int(round(y_ind)))

		### if indices out of range
		# flipped indices range: ([0,5], [0,2])
		move_forward = False
		if ind[0]<0 or ind[0]>5:
			ind = (temp_cur_w_ind[0], int(round(y_ind)))
			move_forward = True
		if ind[1]<0 or ind[1]>2:
			ind = (int(round(x_ind)), temp_cur_w_ind[1])
			move_forward = True
		if ind[0]==5 and ind[1]==2:
			ind = (temp_cur_w_ind[0], temp_cur_w_ind[1])


		# next waypoint info
		if update_w_ind:
			self.set_cur_w_ind(ind)  						
		next_pos = self.waypoints[ind[0]][ind[1]][:3]	# only get position
		next_quat = self.eulerToQuat(heading)[3:] 		# only care about orientation

		# if move_forward:
		# 	if update_w_ind: # no wall detected (update_w_index only true at no wall)
		# 		print('move_forward')
		# 		# pos = self.getPoseGlobal([x_pose, y_pose, heading], th=heading, l=0.25)
		# 	else: 
		# 		print('move_backward')
		# 		# pos = self.getPoseGlobal([x_pose, y_pose, heading], th=heading, l=-0.25)	
		# 	next_pos[0] = pos[0]
		# 	next_pos[1] = pos[1]
		
		waypoint = np.concatenate((next_pos, next_quat))
		waypoint = waypoint.tolist()
		
		print('next waypoint', waypoint)
		return waypoint, heading

	def getPoseGlobal(self,  pos, th, l=0.05):
		p0 = pos
		T = [[np.cos(p0[2]), -np.sin(p0[2]), p0[0]],
			[np.sin(p0[2]), np.cos(p0[2]), p0[1]],
			[0, 0, 1 ]
			]

		p =  np.array([l*np.cos(th), l*np.sin(th), 1]).T
		new_pos = np.matmul(T, p)

		return new_pos

	def classifyImage(self, image):
		"""Classifies the image using KNN

		Args:
			image (np array): rgb image of traffic sign to classifiy

		Returns:
			[string]: calssification result
		"""
		result = self.img_process.classifyImage(image)[0]
		# processed_img = self.img_process.classifyImage(image)[-1]
		# ros_img = CompressedImage()
		# ros_img.data = processed_img

		# self.pub_maskimg.pub(ros_img)

		print('detected: ', result)
		return result

	def doAtWaypoint(self):
		"""Things to do once at waypoint.
		"""
		print('calling doAtWaypoint')
		# assert self.nav_status == 3, 'Nav status not 3 while at waypoint'
		# self.waypointSanityCheck()


		is_wall_infront,_,min_wall_ang = self.isWall(ang=0)
		traffic_sign = 'empty'
		
		if is_wall_infront:
			print('is wall')
			### get rotation ROS waypoint to publish to Nav.
			traffic_sign = self.classifyImage(self.image_np)
			ros_waypoint, desired_yaw = self.getNextWaypoint(traffic_sign, update_w_ind=False) 
			waypoint = self.eulerToQuat(desired_yaw)  
			ros_waypoint = self.getROSNavWaypoint(waypoint)
		
		else:
			print('no wall')
			### get the waypoint in front of the robot
			waypoint, _ = self.getNextWaypoint(traffic_sign)			 # next waypoint in the same heading
			ros_waypoint = self.getROSNavWaypoint(waypoint)

		if traffic_sign != 'goal':
			### publish the next waypoint
			print('publishing nav goal*********************************')
			self.pub_nav.publish(ros_waypoint)
			print("wait for status to update")
			sleep(5.0)
			self.nav_reject_first_status = "No"
		
		else:
			print("\n\n\nReached Goal*******************************")
			sleep(100.0)

	#########################################################
	################## Auxiliary routines ###################
	#########################################################
	def getROSNavWaypoint(self, waypoint):
		"""Generates a waypoint to be passed to the navigation stack

		Args:
			waypoint (list): waypoint quaternion 

		Returns:
			[PoseStamped]: waypoint for the nav. stack
		"""
		position = waypoint[0:3]
		orientation = waypoint[3:]

		waypoint = PoseStamped()
		waypoint.pose.position.x = position[0]
		waypoint.pose.position.y = position[1]
		
		waypoint.pose.orientation.x = orientation[0]
		waypoint.pose.orientation.y = orientation[1]
		waypoint.pose.orientation.z = orientation[2]
		waypoint.pose.orientation.w = orientation[3]

		waypoint.header.frame_id = "map"
		waypoint.header.stamp = rospy.get_rostime()
		
		return waypoint
	
	def calcRobotPose(self):
		"""get the pose of the robot in global frame

		Returns:
			tuple: robot pose 
		"""
		robot_pose = self.get_robot_pose()

		x = robot_pose.pose.pose.position.x
		y = robot_pose.pose.pose.position.y

		### calculate yaw
		q1 = robot_pose.pose.pose.orientation.x
		q2 = robot_pose.pose.pose.orientation.y
		q3 = robot_pose.pose.pose.orientation.z
		q4 = robot_pose.pose.pose.orientation.w

		siny_cosp = 2 * (q4 * q3 + q1 * q2)
		cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
		yaw = np.arctan2(siny_cosp, cosy_cosp)

		return (x, y, yaw)
	
	def eulerToQuat(self, yaw):
		"""Converts Euler angle to Quat. for the robot inplace rotation

		Args:
			yaw (int): new yaw angle

		Returns:
			list: the quaternion representation of the current robot pose with new yaw
		"""
		next_pose = self.get_robot_pose()

		### get the pitch and roll of the robot 
		q = next_pose.pose.pose.orientation
		#  roll (x-axis rotation)
		sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
		cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
		roll = np.arctan2(sinr_cosp, cosr_cosp);

		#  pitch (y-axis rotation)
		sinp = 2 * (q.w * q.y - q.z * q.x);
		if (np.abs(sinp) >= 1):
			pitch = np.sign(sinp) * np.pi/2.0
		else:
			pitch = np.arcsin(sinp)

		### conver Euler to quat
		cy = np.cos(yaw * 0.5);
		sy = np.sin(yaw * 0.5);
		cp = np.cos(pitch * 0.5);
		sp = np.sin(pitch * 0.5);
		cr = np.cos(roll * 0.5);
		sr = np.sin(roll * 0.5);
		
		qw = cr * cp * cy + sr * sp * sy;
		qx = sr * cp * cy - cr * sp * sy;
		qy = cr * sp * cy + sr * cp * sy
		qz = cr * cp * sy - sr * sp * cy;

		x = next_pose.pose.pose.position.x
		y = next_pose.pose.pose.position.y
		z = next_pose.pose.pose.position.z

		pose_array = [x,y,z, qx,qy,qz,qw]

		return pose_array

	def set_cur_w_ind(self, ind):
		print('Updated cur_ind old={} | new={}'.format(self.cur_w_ind , ind))
		self.cur_w_ind = deepcopy(ind)
	
	def get_cur_w_ind(self):
		return deepcopy(self.cur_w_ind)
	
	def set_robot_pose(self, pose):
		self.robot_pose = deepcopy(pose)
	
	def get_robot_pose(self):
		return deepcopy(self.robot_pose)
		
	
	#########################################################
	#################### The Callbacks ######################
	#########################################################
	def callbackCam(self, data):
		"""streams the images from the rpi camera

		Args:
			data (CompressedImage): rpi camera data from the topic
		"""
		np_arr = np.fromstring(data.data, np.uint8)
		self.image_np = cv.imdecode(np_arr, cv.IMREAD_COLOR)

	def callbackLidar(self, data):
		"""streams the Lidar readings and returns the closest point

		Args:
			data (LaserScan): the lidar data
		"""
		self.lidar_arr = data
	
	def callbackAMCLPose(self, data):
		"""Current robot pose. Only works when in navigation mode.

		Args:
			data (PoseWithCovarianceStamped): current robot pose
		"""
		self.set_robot_pose(data)
		if self.robot_pose is None:
			assert False,'huge Error****************'
	
	################################################
	################ The main alg. #################
	################################################
	def callbackNav(self, data):
		"""Callback for Navigation stack. Publishes the correct waypoint

		Args:
			data (GoalStatusArray): the status from the navigation package
		"""	
		# when new checkpoint comes, then only publish new waypoint 
		if self.nav_reject_first_status != "Yes":
			stat = data.status_list
			self.nav_status = int(stat[-1].status)
		
		if self.nav_checkpoint != self.nav_previous_checkpoint: # only executes when at a waypoint
			print('Robot at waypoint*******************************')
			self.doAtWaypoint()
			
		self.nav_previous_checkpoint = self.nav_checkpoint

		# if reached a waypoint
		if self.nav_status == 3: 
			print('\n\n\n****************************************reached waypoint:{}!'.format(self.cur_w_ind))
			self.nav_checkpoint += 1			
			self.nav_status = None
			self.nav_reject_first_status = "Yes"
			sleep(1.0)
	
	def pubRobot(self, vel, omega):
		"""Publish command velocity to robot

		Args:
			vel (float): velocity input
			omega (float): angular velocity input
		"""
		pose = Twist()
		pose.angular.z = omega
		pose.linear.x = vel
		self.pub_vel.publish(pose)


def nav_waypoint():    
	"""Runs the navigation code
	"""
	rospy.init_node('gryffindor_final_demo', anonymous=True)

	zero_vel =  int(sys.argv[1])
	if zero_vel > 0:
		print('zeroing the velocity')
		pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size = 5)
		pose = Twist()
		pose.angular.z = 0
		pose.linear.x = 0
		oldtime = time()
		while (time() - oldtime) < 5:
			pub_vel.publish(pose)
		print('published zero')

	else:
		# run the navigator
		maze_navigator = MazeNavigator()

   	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()
 
if __name__ == '__main__':
	nav_waypoint()
