#include <Eigen/Dense>

#include <ros/ros.h>
#include "ros/param.h"
#include <std_msgs/Float32.h>
#include <coverage/Camera.h>
#include <coverage/Mav.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Bool.h>

#include "Gimbal_vel_ctrl.h"

// class Target_EST_FeedBack
// {
// private:
//     ros::Subscriber target_pose_sub;
//     ros::Subscriber target_twist_sub;
//     ros::Subscriber isTargetEst_sub;
//     geometry_msgs::Pose target_est_pose;
//     geometry_msgs::Twist target_est_twist;
    
// public:
//     bool estimating;
//     Target_EST_FeedBack(ros::NodeHandle& nh_)
//     {   
//         target_pose_sub = nh_.subscribe<geometry_msgs::PoseStamped>("THEIF/pose", 10, &Target_EST_FeedBack::est_pose_cb, this);
//         target_twist_sub = nh_.subscribe<geometry_msgs::TwistStamped>("THEIF/twist", 10, &Target_EST_FeedBack::est_twist_cb, this);
//         isTargetEst_sub = nh_.subscribe<std_msgs::Bool>("THEIF/isTargetEst", 10, &Target_EST_FeedBack::isTargetEst_cb, this);
//     }
//     void est_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg){target_est_pose = msg->pose;}
//     void est_twist_cb(const geometry_msgs::TwistStamped::ConstPtr& msg){target_est_twist = msg->twist;}
//     void isTargetEst_cb(const std_msgs::Bool::ConstPtr& msg)
//     {
//         estimating = msg->data;
//     }
//     geometry_msgs::Pose getPose(){return target_est_pose;}
//     geometry_msgs::Twist getTwist(){return target_est_twist;}
// };


int main(int argc, char** argv)
{
    ros::init(argc, argv, "Gimbal_Track");
    ros::NodeHandle nh;
    ros::Rate rate(100);

    ros::Publisher gimbalVel_roll_pub = nh.advertise<std_msgs::Float32>("gimbal/roll/cmd_joint_velocity", 10);
    ros::Publisher gimbalVel_pitch_pub = nh.advertise<std_msgs::Float32>("gimbal/pitch/cmd_joint_velocity", 10);
    ros::Publisher gimbalVel_yaw_pub = nh.advertise<std_msgs::Float32>("gimbal/yaw/cmd_joint_velocity", 10);
    

    int Self_ID, Target_ID;
    ros::param::get("Self_ID", Self_ID);
    ros::param::get("Target_ID", Target_ID);
    
    std_msgs::Float32 gimbal_roll_cmd;
    std_msgs::Float32 gimbal_pitch_cmd;
    std_msgs::Float32 gimbal_yaw_cmd;

    MAV self(nh, Self_ID); // For UAV
    MAV target(nh, "uav", Target_ID); // For Target
    

    Camera cam(nh, true); // Gimbal for true, normal camera for no
    Gimbal_vel_ctrl gimbalVCtrl;
    // Target_EST_FeedBack target_est(nh);

    MAV_eigen target_eigen = mavMsg2Eigen(target);
    MAV_eigen self_eigen = mavMsg2Eigen(self);

    
    for(int i=0; i<20; i++)
    {
        rate.sleep();
		ros::spinOnce();
    }
    // initialize
    while(ros::ok())
    {
        std::cout << "<MAV_" << Self_ID << ">:\n";
        gimbal_roll_cmd.data = gimbalVCtrl.pControl(0, cam.Roll(), 2);
        gimbal_pitch_cmd.data = gimbalVCtrl.pControl(0, cam.Pitch(), 1);
        gimbal_yaw_cmd.data = gimbalVCtrl.pControl(0, cam.Yaw(), 1);
        
        gimbalVel_roll_pub.publish(gimbal_roll_cmd);
        gimbalVel_pitch_pub.publish(gimbal_pitch_cmd);
        gimbalVel_yaw_pub.publish(gimbal_yaw_cmd);
        std::cout << "Angle initializing\n";
        // if(abs(gimbal_roll_cmd.data) < 0.07 && abs(gimbal_pitch_cmd.data) < 0.07 && abs(gimbal_yaw_cmd.data) < 0.07)
        if(true)
        {
            std::cout << "Angle initialized\n";
            break;
        }
        rate.sleep();
		ros::spinOnce();
    }
    
    while(ros::ok())
	{
        std::cout << "<MAV_" << Self_ID << ">:\n";
        // target_est.estimating = false;
        // if(target_est.estimating)
        // {
        //     target.setPose(target_est.getPose());
        //     target.setTwist(target_est.getTwist());
        //     std::cout << "Vision tracking mode\n";
        // }
        // else
        std::cout << "Ground truth tracking mode\n";

        target_eigen = mavMsg2Eigen(target);
        self_eigen = mavMsg2Eigen(self);
    
        gimbalVCtrl.setTargetData(target_eigen);
        gimbalVCtrl.setSelfData(self_eigen);
        gimbalVCtrl.setCamera(cam);
        gimbalVCtrl.targetTrackingControl(10);
        gimbal_roll_cmd.data = gimbalVCtrl.pControl(0, cam.Roll(), 1);
        gimbal_pitch_cmd.data = gimbalVCtrl.getTiltRate();
        gimbal_yaw_cmd.data = gimbalVCtrl.getPanRate();

        cout << "pitch cmd: " << gimbal_pitch_cmd.data << endl;
        cout << "yaw cmd: " << gimbal_yaw_cmd.data << endl;

        
        if(abs(gimbal_yaw_cmd.data) >= 5)
            gimbal_yaw_cmd.data *= 5/gimbal_yaw_cmd.data;

        if(abs(gimbal_pitch_cmd.data) >= 5)
            gimbal_pitch_cmd.data *= 5/gimbal_yaw_cmd.data;

        gimbalVel_roll_pub.publish(gimbal_roll_cmd);
        gimbalVel_pitch_pub.publish(gimbal_pitch_cmd);
        gimbalVel_yaw_pub.publish(gimbal_yaw_cmd);

		rate.sleep();
		ros::spinOnce();
	}

    return 0;
}