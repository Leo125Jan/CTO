import rospy
import ros_numpy
import numpy as np
import open3d as o3d
import math
from sensor_msgs.msg import LaserScan
import laser_geometry.laser_geometry as lg
from cvxopt import matrix, solvers
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PointStamped


class ConstraintGenerator:
    def __init__(self):
        self.pcl_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.pc_cb)
        self.pos_sub = rospy.Subscriber("/estimator/imu_propagate", Odometry ,self.pos_cb)
        self.rot_sub = rospy.Subscriber("/estimator/rot_state", PointStamped, self.rot_cb)
        self.path_sub = rospy.Subscriber("/path_cmd", Twist,self.path_cmd_cb)
        self.husky_cmd_pub = rospy.Publisher("/cmd_vel", Twist ,queue_size=100)
        self.rate = rospy.Rate(20)

        self.pcl = None
        self.safe_dis = 1

        solvers.options['show_progress'] = False
        self.P = matrix(np.identity(2))
        self.Q = matrix(np.zeros(2))
        self.G = None
        self.H = None
        self.yaw = 0
        self.pos = [0.0,0.0,0.0]
        self.last_pos = [0.0,0.0,0.0]
        self.path_cmd =np.array( [0.0,0.0,0.0,0.0,0.0,0.0])
        self.husky_cmd = Twist()
        self.path_ready = False
        self._initialize()

    def pc_cb(self, data):
        xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        self.pcl = xyz

    def rot_cb(self, msg):
        self.yaw = msg.point.z/180*np.pi #heading angle

    def path_cmd_cb(self,data):
        self.path_ready = True
        
        self.path_cmd[0]=data.linear.x   #x-axis velocity(heading)
        self.path_cmd[1]= data.angular.z #y-axis velocity
        #self.path_cmd[2]=data.linear.z  #empty
        #self.path_cmd[3]=data.angular.x #empty
        #self.path_cmd[4]=data.angular.y #empty
        self.path_cmd[5]=data.angular.z #angular velocity on yaw axis

    def pos_cb(self,data):
        self.pos[0] = data.pose.pose.position.x
        self.pos[1] = data.pose.pose.position.y
        self.pos[2] = data.pose.pose.position.z

    def contraint_solver(self):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(self.pcl)
        downpcl = pcl.voxel_down_sample(voxel_size=0.05)
        pcl = np.asarray(downpcl.points)
        pcl = pcl[np.abs(pcl[:, 0]) < 5]
        pcl = pcl[np.abs(pcl[:, 1]) < 5]
        pcl = pcl[np.abs(pcl[:, 2]) < 1]
        self.pcl=-1*self.pcl
        dis_sum_square=np.square(self.pcl).sum(axis=1)
        g = np.vstack([np.array([2*-(self.pcl[:, 0]+0.5*math.cos(self.yaw))*math.cos(self.yaw)+2*-(self.pcl[:, 1]+0.5*math.sin(self.yaw))*math.sin(self.yaw)]), np.array([-2*-(self.pcl[:, 0]+0.5*math.cos(self.yaw))*0.5*math.sin(self.yaw)+2*-(self.pcl[:, 1]+0.5*math.sin(self.yaw))*0.5*math.cos(self.yaw)])]).T
        h = (dis_sum_square - (self.safe_dis * self.safe_dis + 0.1) * np.ones(len(self.pcl)))*2
              
        self.Q = matrix(-0.5*self.path_cmd[0:2],tc='d')
        self.G = matrix(g,tc='d')
        self.H = matrix(h,tc='d')
        sol=solvers.coneqp(self.P, self.Q, self.G, self.H)
        u_star = sol['x']
        error_yaw=math.atan(u_star[1]/u_star[0])
        yaw_rate = np.arctan(np.inner(np.array([-math.sin(self.yaw), math.cos(self.yaw)]),np.array([u_star[0],u_star[1]]))/np.inner(np.array([math.cos(self.yaw),math.sin(self.yaw)]),np.array([u_star[0],u_star[1]])))
        k_v = 1
        kp_yaw = 0.8
        #print("u*: ", u_star)
        self.husky_cmd.linear.x = k_v *u_star[0]#np.inner(np.array([math.cos(self.yaw),math.sin(self.yaw)]),np.array([u_star[0],u_star[1]]))
        self.husky_cmd.angular.z = kp_yaw *u_star[1]

        #print("complete optimization ")
    def _initialize(self):
        while self.pcl is None:
            print("No point cloud info")
            self.rate.sleep()

    def process(self):
        #print("Ready to solve qp")
        if self.path_ready:
            self.contraint_solver()
            self.husky_cmd_pub.publish(self.husky_cmd)

if __name__ == "__main__":
    rospy.init_node("constraint_generator_node")
    cg = ConstraintGenerator()
    while not rospy.is_shutdown():
        #cg.process()
        try : cg.process()
        except: print("error")
        rospy.sleep(0.01)