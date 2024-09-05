#include "Camera.h"

Camera::Camera()
{
    fx_ = 1280.7847335087872;
    fy_ = 1280.7847335087872;
    cx_ = 640.5;
    cy_ = 360.5;
    lx_ = 1280;
    ly_ = 720;

	t_b2c.setZero();
	R_b2c.setIdentity();
}

Camera::Camera(ros::NodeHandle &nh_, bool gimbal)
{
    fx_ = 1280.7847335087872;
    fy_ = 1280.7847335087872;
    cx_ = 640.5;
    cy_ = 360.5;
    if(gimbal)
    {
        nh = nh_;
        jointState_sub = nh.subscribe<sensor_msgs::JointState>("gimbal/joint_states", 5, &Camera::jointState_cb, this);

        t_b2c << 0.041, 0, -0.162;
        R_b2m << 1, 0, 0,
                0, -1, 0,
                0, 0, -1;
        R_t2c << 0, 1, 0,
                0, 0, 1,
                1, 0, 0;
    }
    else
    {   
        t_b2c<< 0.1, 0, 0; 
        R_b2c<< 0, 1, 0,
                0, 0, 1,
                1, 0, 0;
    }

}

Camera::~Camera(){}

void Camera::jointState_cb(const sensor_msgs::JointState::ConstPtr& msg)
{
    roll = msg->position[0];
    pitch = msg->position[1];
    yaw = msg->position[2];

    rate_roll = msg->velocity[0];
    rate_pitch = msg->velocity[1];
    rate_yaw = msg->velocity[2];

    R_m2p << cos(yaw), sin(yaw), 0,
            -sin(yaw), cos(yaw), 0,
            0, 0, 1;
    R_p2t << cos(pitch), 0, -sin(pitch),
            0, 1, 0,
            sin(pitch), 0, cos(pitch);

    R_b2c = R_t2c*R_p2t*R_m2p*R_b2m;

    omega_b2c(0) = rate_pitch;
    omega_b2c(1) = rate_yaw*cos(pitch);
    omega_b2c(2) = -rate_yaw*sin(pitch);
}


void Camera::setParameters(double f_x, double f_y, double c_x, double c_y)
{
    fx_ = f_x;
    fy_ = f_y;
    cx_ = c_x;
    cy_ = c_y;
}

void Camera::setRelativePose(Eigen::Matrix3d R_B2C, Eigen::Vector3d t_B2C)
{
    R_b2c = R_B2C;
    t_b2c = t_B2C;
}

double Camera::fx(){return fx_;}
double Camera::fy(){return fy_;}
double Camera::cx(){return cx_;}
double Camera::cy(){return cy_;}
double Camera::lx(){return lx_;}
double Camera::ly(){return ly_;}
double Camera::Roll(){return roll;}
double Camera::Pitch(){return pitch;}
double Camera::Yaw(){return yaw;}

Eigen::Vector3d Camera::t_B2C(){return t_b2c;}
Eigen::Matrix3d Camera::R_B2C(){return R_b2c;}
Eigen::Vector3d Camera::omega_B2C(){return omega_b2c;}