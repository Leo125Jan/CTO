#include "Gimbal_vel_ctrl.h"

Gimbal_vel_ctrl::Gimbal_vel_ctrl()
{
    m.setZero();
    x.setZero();

    L1_hat.setZero(2, 2);
    L2.setZero(2, 6);
    vel.setZero(6);
}
Gimbal_vel_ctrl::~Gimbal_vel_ctrl(){}

void Gimbal_vel_ctrl::setTargetData(MAV_eigen target)
{
    t = target;
}
void Gimbal_vel_ctrl::setSelfData(MAV_eigen self)
{
    s = self;
}
void Gimbal_vel_ctrl::setCamera(Camera camera)
{
    cam = camera;
}

double Gimbal_vel_ctrl::pControl(double desired, double current, double kp)
{
    double err = desired - current;

    if(err>M_PI)
        err = err - 2*M_PI;
    else if(err<-M_PI)
        err = err + 2*M_PI;

    return kp*err;
}

void Gimbal_vel_ctrl::targetTrackingControl(double gain)
{   
    R_w2c = cam.R_B2C()*s.R_w2b;
    r_tc_c = R_w2c*(t.r - s.r_c);

    m = cam.R_B2C()*cam.t_B2C();
    x(0) = r_tc_c(0)/r_tc_c(2);
    x(1) = r_tc_c(1)/r_tc_c(2);
    x(2) = 1/r_tc_c(2);

    V_t_c = R_w2c*t.v;
    V_s_c = R_w2c*s.v;

    L1_hat << x(0)*x(1), -(1 + x(0)*x(0)),
                (1 + x(1)*x(1)), -x(0)*x(1);
    
    L2 << -x(2), 0, x(0)*x(2), x(0)*(x(1) + m(1)*x(2)), -(1 + x(0)*x(0) + x(2)*(m(2) + m(0)*x(0))), x(1) + m(1)*x(2),
            0, -x(2), x(1)*x(2), 1 + x(1)*x(1) + x(2)*(m(2) + m(1)*x(1)), -x(1)*(x(0)+m(0)*x(2)), -x(0) - m(0)*x(2);

    vel << V_s_c - V_t_c, R_w2c*s.omega_c;
    //vel << Eigen::Vector3d::Zero(), R_w2c*s.omega_c;

    e << x(0), x(1);

    omega = -L1_hat.inverse()*L2*vel - gain*L1_hat.inverse()*e;

    rate_tilt = omega(0);
    rate_pan = omega(1)/cos(cam.Pitch());


    //std::cout << "r_tc_b:\n" << s.R_w2b*(t.r - s.r_c)  << "\n\n";
    //std::cout << "r_tc_c:\n" << r_tc_c  << std::endl;
    //std::cout << "\n m:\n" << m << "\n";
    // std::cout << "\n x:\n" << x << "\n";
    //std::cout << "rate_tilt: " << rate_tilt << " rate_pan: " << rate_pan << std::endl;
}

double Gimbal_vel_ctrl::getTiltRate(){return rate_tilt;}
double Gimbal_vel_ctrl::getPanRate(){return rate_pan;}