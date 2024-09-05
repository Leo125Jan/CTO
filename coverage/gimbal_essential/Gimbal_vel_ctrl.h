#ifndef GIMBAL_VEL_CTRL
#define GIMBAL_VEL_CTRL
#pragma once
#include <Eigen/Dense>

#include <coverage/Mav.h>
#include <coverage/Camera.h>

class Gimbal_vel_ctrl
{
private:
    MAV_eigen t;
    MAV_eigen s;

    Eigen::Vector3d m;
    Eigen::Vector3d x;
    Eigen::Vector3d r_tc_c;
    Eigen::Matrix3d R_w2c;
    Eigen::Vector3d V_t_c;
    Eigen::Vector3d V_s_c;
    Eigen::MatrixXd L1_hat;
    Eigen::MatrixXd L2;
    Eigen::VectorXd vel;
    Eigen::Vector2d e;
    Eigen::Vector2d omega;

    double rate_tilt;
    double rate_pan;
    Camera cam;
public:
    Gimbal_vel_ctrl();
    ~Gimbal_vel_ctrl();
    void setTargetData(MAV_eigen target);
    void setSelfData(MAV_eigen self);
    void targetTrackingControl(double gain);
    void setCamera(Camera camera);
    double pControl(double desired, double current, double kp);
    double getTiltRate();
    double getPanRate();
};





#endif