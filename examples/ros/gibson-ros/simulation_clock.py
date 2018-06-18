#!/usr/bin/python
import rospy
from std_msgs.msg import Int64

def talker():
    pub = rospy.Publisher('/gibson_ros/sim_clock', Int64, queue_size=10)
    rospy.init_node('gibson_ros_clock')
    rate = rospy.Rate(1000) # 1000hz
    while not rospy.is_shutdown():
        pub.publish(rospy.get_time())
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
