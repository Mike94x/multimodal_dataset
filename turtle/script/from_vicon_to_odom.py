#!/usr/bin/python3
import turtle
import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import math
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
import tf
from turtle.msg import SafeVicon


vic_x=3 
vic_y=2.8

pose = TransformStamped()
def vicon_cb(data):
    global pose
    # print("pose", pose)
    pose = data

def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y

	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))

	return X, Y, Z

rospy.init_node('vicon_conversion_to_odom')

vicon_sub = rospy.Subscriber('/vicon/turtlebot3/turtlebot3', SafeVicon, vicon_cb, queue_size=100)
odom_pub = rospy.Publisher('turtlebot3/odom', Odometry, queue_size=100)

rate = rospy.Rate(50.0)
counter = 0
x = 0.
y = 0.

dt = 1./50.

while not rospy.is_shutdown():
    position = pose.transform.translation
    rotation = pose.transform.rotation

    (v_roll,v_pitch,v_yaw) = quaternion_to_euler_angle(rotation.w, rotation.x , rotation.y, rotation.z)
    v_phi = float((v_roll))
    v_theta = float((v_pitch))
    v_psi = float((v_yaw))
    
    x = position.x+vic_x #correggo la posizione letta dal vicon in modo tale che sia sempre positiva (sposto il centro del sdr nell'angolo opposto alla porta)
    y = position.y+vic_y
    z = position.z

    yaw = math.radians(v_psi)

    if counter > 0:
        vel_x_world = (x - x_prev) / dt
        vel_y_world = (y - y_prev) / dt

        x_prev = x
        y_prev = y

        twist_x = math.cos(yaw) * vel_x_world + math.sin(yaw) * vel_y_world
        twist_y = math.cos(yaw) * vel_y_world - math.sin(yaw) * vel_x_world

        odom = Odometry()
        odom.header.frame_id = 'turtlebot3/odom_vicon'
        odom.child_frame_id = 'turtlebot3/base_footprint2'
        odom.header.stamp = rospy.Time.now()

        odom.pose.pose.position.x = round(position.x,4)
        odom.pose.pose.position.y = round(position.y,4)
        odom.pose.pose.position.z = round(position.z,4)

        odom.pose.pose.orientation.x = round(rotation.x,4)
        odom.pose.pose.orientation.y = round(rotation.y,4)
        odom.pose.pose.orientation.z = round(rotation.z,4)
        odom.pose.pose.orientation.w = round(rotation.w,4)

        odom.twist.twist.linear.x = round(twist_x,4)
        odom.twist.twist.linear.y = round(twist_y,4)
        odom.twist.twist.linear.z = round((z - z_prev) / dt,4)
        z_prev = round(z,4)

        odom.twist.twist.angular.x = 0.
        odom.twist.twist.angular.y = 0.
        odom.twist.twist.angular.z = 0.



        odom_pub.publish(odom)

        br = tf.TransformBroadcaster()
        br.sendTransform((x,y,z),[rotation.x, rotation.y, rotation.z,rotation.w],rospy.Time.now(), "turtlebot3/base_footprint2","turtlebot3/odom_vicon")

    else:
        x_prev = x
        y_prev = y
        z_prev = z
        counter += 1



    rate.sleep()

