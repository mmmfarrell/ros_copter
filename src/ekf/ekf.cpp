#include "ekf/ekf.h"
#include <stdio.h>

using namespace std;

namespace ekf
{

mocapFilter::mocapFilter() :
  nh_(ros::NodeHandle()),
  nh_private_(ros::NodeHandle("~/ekf"))
{
  // retrieve params
  nh_private_.param<double>("inner_loop_rate", inner_loop_rate_, 400);
  nh_private_.param<double>("publish_rate", publish_rate_, 400);
  nh_private_.param<double>("alpha", alpha_, 0.2);
  ros_copter::importMatrixFromParamServer(nh_private_, x_hat_, "x0");
  ros_copter::importMatrixFromParamServer(nh_private_, P_, "P0");
  ros_copter::importMatrixFromParamServer(nh_private_, Q_, "Q0");
  ros_copter::importMatrixFromParamServer(nh_private_, R_IMU_, "R_IMU");
  ros_copter::importMatrixFromParamServer(nh_private_, R_Mocap_, "R_Mocap");
  ros_copter::importMatrixFromParamServer(nh_private_, R_GPS_, "R_GPS");
  ros_copter::importMatrixFromParamServer(nh_private_, R_Mag_, "R_Mag");

  // Setup publishers and subscribers
  imu_sub_ = nh_.subscribe("imu/data", 1, &mocapFilter::imuCallback, this);
  mocap_sub_ = nh_.subscribe("mocap", 1, &mocapFilter::mocapCallback, this);
  gps_sub_ = nh_.subscribe("gps/data", 1, &mocapFilter::gpsCallback, this);
  mag_sub_ = nh_.subscribe("shredder/mag/data", 1, &mocapFilter::magCallback, this);
  estimate_pub_ = nh_.advertise<nav_msgs::Odometry>("estimate", 1);
  bias_pub_ = nh_.advertise<sensor_msgs::Imu>("estimate/bias", 1);
  is_flying_pub_ = nh_.advertise<std_msgs::Bool>("is_flying", 1);
  predict_timer_ = nh_.createTimer(ros::Duration(1.0/inner_loop_rate_), &mocapFilter::predictTimerCallback, this);
  publish_timer_ = nh_.createTimer(ros::Duration(1.0/publish_rate_), &mocapFilter::publishTimerCallback, this);

  lat0_ = 0;
  lon0_ = 0;
  alt0_ = 0;
  gps_count_ = 0;

  x_hat_.setZero();
  flying_ = false;
  ROS_INFO("Done");
  return;
}

void mocapFilter::publishTimerCallback(const ros::TimerEvent &event)
{
  publishEstimate();
  return;
}

void mocapFilter::predictTimerCallback(const ros::TimerEvent &event)
{
  if(flying_){
    predictStep();
  }
  return;
}


void mocapFilter::imuCallback(const sensor_msgs::Imu msg)
{
  if(!flying_){
    if(fabs(msg.linear_acceleration.z) > 10.0){
      ROS_WARN("Now flying");
      flying_ = true;
      std_msgs::Bool flying;
      flying.data = true;
      is_flying_pub_.publish(flying);
      previous_predict_time_ = ros::Time::now();
    }
  }
  if(flying_){
    updateIMU(msg);
  }
  return;
}


void mocapFilter::mocapCallback(const geometry_msgs::TransformStamped msg)
{
  if(!flying_){
    ROS_INFO_THROTTLE(1,"Not flying, but motion capture received, estimate is copy of mocap");
    initializeX(msg);
  }else{
    updateMocap(msg);
  }
  return;
}

void mocapFilter::gpsCallback(const fcu_common::GPS msg)
{
  if(!flying_){
    ROS_INFO_THROTTLE(1,"Not flying, but GPS signal received");
    lat0_ = (gps_count_*lat0_ + msg.latitude)/(gps_count_ + 1);
    lon0_ = (gps_count_*lon0_ + msg.longitude)/(gps_count_ + 1);
    alt0_ = (gps_count_*alt0_ + msg.altitude)/(gps_count_ + 1);
    gps_count_++;
  }else{
    if(msg.fix){
      updateGPS(msg);
    }
  }
}

void mocapFilter::magCallback(const sensor_msgs::MagneticField msg)
{
  if(gps_recieved){
      updateMag(msg);
  }
}

void mocapFilter::initializeX(geometry_msgs::TransformStamped msg)
{
  ROS_INFO_ONCE("ekf initialized");
  tf::Transform measurement;
  double roll, pitch, yaw;
  tf::transformMsgToTF(msg.transform, measurement);
  tf::Matrix3x3(measurement.getRotation()).getRPY(roll, pitch, yaw);
  x_hat_(PN) =  measurement.getOrigin().getX(); // NWU to NED
  x_hat_(PE) = -measurement.getOrigin().getY();
  x_hat_(PD) = -measurement.getOrigin().getZ();
  x_hat_(PHI) = roll;
  x_hat_(THETA) = -pitch;
  x_hat_(PSI) = -yaw;
  return;
}


void mocapFilter::predictStep()
{
  ros::Time now = ros::Time::now();
  double dt = (now-previous_predict_time_).toSec();
  x_hat_ = x_hat_ + dt*f(x_hat_);
  Eigen::Matrix<double, NUM_STATES, NUM_STATES> A = dfdx(x_hat_);
  P_ = P_ + dt*(A*P_ + P_*A.transpose() + Q_);
  previous_predict_time_ = now;
  return;
}


void mocapFilter::updateIMU(sensor_msgs::Imu msg)
{
  //ROS_INFO_STREAM("Update IMU");
  double phi(x_hat_(PHI)), theta(x_hat_(THETA)), psi(x_hat_(PSI));
  double alpha_x(x_hat_(AX)), alpha_y(x_hat_(AY)), alpha_z(x_hat_(AZ));
  double ct = cos(theta);
  double cp = cos(phi);
  double st = sin(theta);
  double sp = sin(phi);
  ax_ = msg.linear_acceleration.x;
  ay_ = msg.linear_acceleration.y;
  az_ = msg.linear_acceleration.z;
  gx_ = msg.angular_velocity.x;
  gy_ = msg.angular_velocity.y;
  gz_ = msg.angular_velocity.z;

  Eigen::Matrix<double, 3, 1> y;
  y << ax_, ay_, az_;

  Eigen::Matrix<double, 3, NUM_STATES> C;
  C.setZero();
  C(0,THETA) = -G*ct;
  C(0,AX) = -1.0;

  C(1,PHI) = G*ct*cp;
  C(1,THETA) = -G*st*sp;
  C(1,AY) = -1.0;

  C(2,PHI) = G*ct*sp;
  C(2,THETA) = G*st*cp;
  C(2,AZ) = -1.0;

  Eigen::Matrix<double, NUM_STATES, 3> L;
  L.setZero();
  L = P_*C.transpose()*(R_IMU_ + C*P_*C.transpose()).inverse();
  P_ = (Eigen::MatrixXd::Identity(NUM_STATES,NUM_STATES) - L*C)*P_;
  x_hat_ = x_hat_ + L*(y - C*x_hat_);
}

void mocapFilter::updateMocap(geometry_msgs::TransformStamped msg)
{
  tf::Transform measurement;
  double roll, pitch, yaw;
  tf::transformMsgToTF(msg.transform, measurement);
  tf::Matrix3x3(measurement.getRotation()).getRPY(roll, pitch, yaw);
  Eigen::Matrix<double, 6, 1> y;
  y << measurement.getOrigin().getX(),
       measurement.getOrigin().getY(),
       measurement.getOrigin().getZ(),
       roll,  pitch,  yaw;
  Eigen::Matrix<double, 6, NUM_STATES> C = Eigen::Matrix<double, 6, NUM_STATES>::Zero();
  C(0,PN) = 1.0;
  C(1,PE) = 1.0;
  C(2,PD) = 1.0;
  C(3,PHI) = 1.0;
  C(4,THETA) = 1.0;
  C(5,PSI) = 1.0;
  Eigen::Matrix<double, NUM_STATES, 6> L;
  L.setZero();
  L = P_*C.transpose()*(R_Mocap_ + C*P_*C.transpose()).inverse();
  P_ = (Eigen::MatrixXd::Identity(NUM_STATES,NUM_STATES) - L*C)*P_;
  x_hat_ = x_hat_ + L*(y - C*x_hat_);
}

void mocapFilter::updateGPS(fcu_common::GPS msg)
{
  //ROS_INFO_STREAM("Update GPS");

  static bool first_time = true;
  if(first_time)
  {
    first_time = false;
    if(gps_count_ < 1)
    {
      ROS_ERROR("First GPS message received without initialization");
      lat0_ = msg.latitude;
      lon0_ = msg.longitude;
      alt0_ = msg.altitude;
      return;
    }
  }
  gps_recieved = true;

  double u = x_hat_(U);
  double v = x_hat_(V);
  lat_ = msg.latitude;
  lon_ = msg.longitude;
  alt_ = msg.altitude;
  vg_ =  msg.speed;
  chi_ = msg.ground_course;

  double dx, dy, dlat, dlon;
  dlat = (lat_ - lat0_);
  dlon = (lon_ - lon0_);
  GPS_to_m(&dlat, &dlon, &dx, &dy);


  Eigen::Matrix<double, 3, 1> y;
  y << dx, dy, -(alt_ - alt0_);//, vg_, chi_;

//  cout << "y = " << y << endl;
//  cout << "dlat = " << dlat << " dlon = " << dlon << endl;

  Eigen::Matrix<double, 3, NUM_STATES> C;
  C.setZero();
  C(0,PN) = 1.0;

  C(1,PE) = 1.0;

  C(2,PD) = 1.0;

//  C(3,U) = u/sqrt(u*u+v*v);
//  C(3,V) = v/sqrt(u*u+v*v);

//  C(4,U) = -v/(u*u+v*v);
//  C(4,V) = u/(u*u+v*v);

  Eigen::Matrix<double, NUM_STATES, 3> L;
  L.setZero();
  L = P_*C.transpose()*(R_GPS_ + C*P_*C.transpose()).inverse();
  P_ = (Eigen::MatrixXd::Identity(NUM_STATES,NUM_STATES) - L*C)*P_;
  x_hat_ = x_hat_ + L*(y - C*x_hat_);
//  ROS_INFO("x = %0.02f\n", x_hat_(PN));
}

void mocapFilter::updateMag(sensor_msgs::MagneticField msg)
{
  static bool first_time = true;
  if(first_time)
  {
      if(gps_recieved){
        ROS_INFO_THROTTLE(1,"Magnetometer online");
        first_time = false;
        bx = .406569;
        by = .0818167;
        bz = .909949;
//        MAGtype_GeoMagneticElements mag = init_magnetic_field(lat0_,lon0_,alt0_);
//        bx = mag.X;
//        by = -mag.Y;
//        bz = -mag.Z;
        cout << endl << "magnetic field" << endl << bx << endl << by << endl << bz << endl << endl;
      }
      else{
          ROS_INFO_THROTTLE(1,"Waiting for GPS to determine local magnetic field");
      }
  }

  double phi(x_hat_(PHI)), theta(x_hat_(THETA)), psi(x_hat_(PSI));
  double ct = cos(theta);
  double cp = cos(phi);
  double cs = cos(psi);
  double st = sin(theta);
  double sp = sin(phi);
  double ss = sin(psi);
  mx_ = msg.magnetic_field.x;
  my_ = msg.magnetic_field.y;
  mz_ = msg.magnetic_field.z;

  Eigen::Matrix<double, 3, 1> y;
  y << mx_,my_,mz_;
    cout << "y = " << y << endl;

  Eigen::Matrix<double, 3, NUM_STATES> C;
  C.setZero();
  C(0,PHI)   =             by*(cp*st*cs + sp*ss)  + bz*(-sp*st*cs + cp*ss);
  C(0,THETA) = -bx*st*cs + by*sp*ct*cs            + bz*cp*ct*cs;
  C(0,PSI)   = -bx*ct*ss + by*(-cp*cs -sp*st*ss)  + bz*(sp*cs - cp*st*ss);

  C(1,PHI)   =             by*(-sp*cs + cp*st*ss) + bz*(sp*cs - cp*st*ss);
  C(1,THETA) = -bx*st*ss + by*sp*ct*ss            + bz*cp*ct*ss;
  C(1,PSI)   =  bx*ct*cs + by*(sp*st*cs - cp*ss)  + bz*(cp*st*cs + sp*ss);

  C(2,PHI)   =             by*cp*ct               - bz*sp*ct;
  C(2,THETA) = -bx*ct    - by*sp*st               + bz*cp*st;

  Eigen::Matrix<double, NUM_STATES, 3> L;
  L.setZero();
  L = P_*C.transpose()*(R_Mag_ + C*P_*C.transpose()).inverse();
  P_ = (Eigen::MatrixXd::Identity(NUM_STATES,NUM_STATES) - L*C)*P_;
  cout << C*x_hat_ << endl;
  x_hat_ = x_hat_ + L*(y - C*x_hat_);
  ROS_INFO("x = %e, %e, %e\n", x_hat_(PHI), x_hat_(THETA), x_hat_(PSI));
}

MAGtype_GeoMagneticElements mocapFilter::init_magnetic_field(double lat, double lon, double alt){
    MAGtype_MagneticModel * MagneticModels[1], *TimedMagneticModel;
    MAGtype_Ellipsoid Ellip;
    MAGtype_CoordSpherical CoordSpherical;
    MAGtype_CoordGeodetic CoordGeodetic;
    MAGtype_Date UserDate;
    MAGtype_GeoMagneticElements GeoMagneticElements;
    MAGtype_Geoid Geoid;
    char* filename = "WMM.COF";
    int NumTerms, nMax = 0;
    int epochs = 1;

    MAG_robustReadMagModels(filename, &MagneticModels, epochs);

    /* Memory allocation */
    if(nMax < MagneticModels[0]->nMax) nMax = MagneticModels[0]->nMax;
    NumTerms = ((nMax + 1) * (nMax + 2) / 2);
    TimedMagneticModel = MAG_AllocateModelMemory(NumTerms); /* For storing the time modified WMM Model parameters */
    cout << "allocate memory" << endl;
    if(MagneticModels[0] == NULL || TimedMagneticModel == NULL)
    {
        MAG_Error(2);
    }

    MAG_SetDefaults(&Ellip, &Geoid); /* Set default values and constants */
    cout << "set defaults" << endl;

    CoordGeodetic.phi = lat;
    CoordGeodetic.lambda = lon;
    CoordGeodetic.HeightAboveGeoid = alt/1000;
    MAG_ConvertGeoidToEllipsoidHeight(&CoordGeodetic, &Geoid);
    time_t t = time(0);
    struct tm * now = localtime( & t );
    UserDate.Year = now->tm_year + 1900;
    UserDate.Month = now->tm_mon + 1;
    UserDate.Day = now->tm_mday;
    cout << "create date" << endl;

    MAG_GeodeticToSpherical(Ellip, CoordGeodetic, &CoordSpherical); /*Convert from geodetic to Spherical Equations: 17-18, WMM Technical report*/
    cout << "convert to spherical" << endl;
    MAG_TimelyModifyMagneticModel(UserDate, MagneticModels[0], TimedMagneticModel); /* Time adjust the coefficients, Equation 19, WMM Technical report */
    cout << "time modify" << endl;
    MAG_Geomag(Ellip, CoordSpherical, CoordGeodetic, TimedMagneticModel, &GeoMagneticElements); /* Computes the geoMagnetic field elements and their time change*/
    cout << "compute geomag" << endl;
    MAG_CalculateGridVariation(CoordGeodetic, &GeoMagneticElements);

    MAG_FreeMagneticModelMemory(TimedMagneticModel);
    MAG_FreeMagneticModelMemory(MagneticModels[0]);

    return GeoMagneticElements;
}


Eigen::Matrix<double, NUM_STATES, 1> mocapFilter::f(const Eigen::Matrix<double, NUM_STATES, 1> x)
{
  double u(x(U)), v(x(V)), w(x(W));
  double phi(x(PHI)), theta(x(THETA)), psi(x(PSI));
  double alpha_z(x(AZ));
  double p(gx_+x(BX)), q(gy_+x(BY)), r(gz_+x(BZ));

  double ct = cos(theta);
  double cs = cos(psi);
  double cp = cos(phi);
  double st = sin(theta);
  double ss = sin(psi);
  double sp = sin(phi);
  double tt = tan(theta);

  // calculate forces - see eqs. 16-19, and 35-37.
  Eigen::Matrix<double, NUM_STATES, 1> xdot;
  xdot.setZero();
  xdot(PN) = ct*cs*u + (sp*st*cs-cp*ss)*v + (cp*st*cs+sp*ss)*w;
  xdot(PE) = ct*ss*u + (sp*st*ss+cp*cs)*v + (cp*st*ss-sp*cs)*w;
  xdot(PD) = -st*u   + sp*ct*v            + cp*ct*w;

  xdot(U) = r*v-q*w - G*st;
  xdot(V) = p*w-r*u + G*ct*sp;
  xdot(W) = q*u-p*v + G*ct*cp + (az_+alpha_z);

  xdot(PHI) = p + sp*tt*q + cp*tt*r;
  xdot(THETA) = q*ct - r*sp;
  xdot(PSI) = q*sp/ct + r*cp/ct;

  // all other states (biases) are constant
  return xdot;
}



Eigen::Matrix<double, NUM_STATES, NUM_STATES> mocapFilter::dfdx(const Eigen::Matrix<double, NUM_STATES, 1> x)
{
  double u(x(U)), v(x(V)), w(x(W));
  double phi(x(PHI)), theta(x(THETA)), psi(x(PSI));
  double p(gx_+x(BX)), q(gy_+x(BY)), r(gz_+x(BZ));

  double ct = cos(theta);
  double cs = cos(psi);
  double cp = cos(phi);
  double st = sin(theta);
  double ss = sin(psi);
  double sp = sin(phi);
  double tt = tan(theta);

  Eigen::Matrix<double, NUM_STATES, NUM_STATES> A;
  A.setZero();
  A(PN,U) = ct*cs;
  A(PN,V) = sp*st*cs-cp*ss;
  A(PN,W) = cp*st*cs+sp*ss;
  A(PN,PHI) = (cp*st*cs+sp*ss)*v + (-sp*st*cs+cp*ss)*w;
  A(PN,THETA) = -st*cs*u + (sp*ct*cs)*v + (cp*ct*cs)*w;
  A(PN,PSI) = -ct*ss*u + (-sp*st*ss-cp*cs)*v + (-cp*st*ss+sp*cs)*w;

  A(PE,U) = ct*ss;
  A(PE,V) = sp*st*ss+cp*cs;
  A(PE,W) = cp*st*ss-sp*cs;
  A(PE,PHI) = (cp*st*ss-sp*cs)*v + (-sp*st*ss-cp*cs)*w;
  A(PE,THETA) = -st*cs*u + (sp*ct*ss)*v + (cp*ct*ss)*w;
  A(PE,PSI) = -ct*ss*u  + (sp*st*cs-cp*ss)*v + (cp*st*cs+sp*ss)*w;

  A(PD,U) = -st;
  A(PD,V) = sp*ct;
  A(PD,W) = cp*ct;
  A(PD,PHI) = cp*ct*v - sp*ct*w;
  A(PD,THETA) = -ct*u - sp*st*v - cp*st*w;

  A(U,V) = r;
  A(U,W) = -q;
  A(U,THETA) = -G*ct;
  A(U,BY) = -w;
  A(U,BZ) = v;

  A(V,U) = -r;
  A(V,W) = p;
  A(V,PHI) = G*ct*cp;
  A(V,THETA) = -G*st*sp;
  A(V,BX) = w;
  A(V,BZ) = -u;

  A(W,U) = q;
  A(W,V) = -p;
  A(W,PHI) = -G*ct*sp;
  A(W,THETA) = -G*st*cp;
  A(W,AZ) = 1.0;
  A(W,BX) = -v;
  A(W,BY) = u;

  A(PHI,PHI) = cp*tt*q - sp*tt*r;
  A(PHI,THETA) = (sp*q + cp*r)/(ct*ct);
  A(PHI,BX) = 1;
  A(PHI,BY) = sp*tt;
  A(PHI,BZ) = cp*tt;

  A(THETA,PHI) = -sp*q -cp*r;
  A(THETA,BY) = cp;
  A(THETA,BZ) = -sp;

  A(PSI,PHI) = (q*cp-r*sp)/ct;
  A(PSI,THETA) = (q*sp+r*cp)*tt/ct;
  A(PSI,BY) = sp/ct;
  A(PSI,BZ) = cp/ct;

  // all other states are zero

  return A;
}

void mocapFilter::publishEstimate()
{
  nav_msgs::Odometry estimate;
  double pn(x_hat_(PN)), pe(x_hat_(PE)), pd(x_hat_(PD));
  double u(x_hat_(U)), v(x_hat_(V)), w(x_hat_(W));
  double phi(x_hat_(PHI)), theta(x_hat_(THETA)), psi(x_hat_(PSI));
  double bx(x_hat_(BX)), by(x_hat_(BY)), bz(x_hat_(BZ));

  tf::Quaternion q;
  q.setRPY(phi, theta, psi);
  geometry_msgs::Quaternion qmsg;
  tf::quaternionTFToMsg(q, qmsg);
  estimate.pose.pose.orientation = qmsg;

  estimate.pose.pose.position.x = pn;
  estimate.pose.pose.position.y = pe;
  estimate.pose.pose.position.z = pd;

  estimate.pose.covariance[0*6+0] = P_(PN, PN);
  estimate.pose.covariance[1*6+1] = P_(PE, PE);
  estimate.pose.covariance[2*6+2] = P_(PD, PD);
  estimate.pose.covariance[3*6+3] = P_(PHI, PHI);
  estimate.pose.covariance[4*6+4] = P_(THETA, THETA);
  estimate.pose.covariance[5*6+5] = P_(PSI, PSI);

  estimate.twist.twist.linear.x = u;
  estimate.twist.twist.linear.y = v;
  estimate.twist.twist.linear.z = w;
  estimate.twist.twist.angular.x = gx_+bx;
  estimate.twist.twist.angular.y = gy_+by;
  estimate.twist.twist.angular.z = gz_+bz;

  estimate.twist.covariance[0*6+0] = P_(U, U);
  estimate.twist.covariance[1*6+1] = P_(V, V);
  estimate.twist.covariance[2*6+2] = P_(W, W);
  estimate.twist.covariance[3*6+3] = 0.05;  // not being estimated
  estimate.twist.covariance[4*6+4] = 0.05;
  estimate.twist.covariance[5*6+5] = 0.05;

  estimate.header.frame_id = "body_link";
  estimate.header.stamp = ros::Time::now();
  estimate_pub_.publish(estimate);

  sensor_msgs::Imu bias;
  bias.linear_acceleration.x = x_hat_(AX);
  bias.linear_acceleration.y = x_hat_(AY);
  bias.linear_acceleration.z = x_hat_(AZ);
  bias.linear_acceleration_covariance[0*3+0] = P_(AX,AX);
  bias.linear_acceleration_covariance[1*3+1] = P_(AY,AY);
  bias.linear_acceleration_covariance[2*3+2] = P_(AZ,AZ);

  bias.angular_velocity.x = x_hat_(BX);
  bias.angular_velocity.y = x_hat_(BY);
  bias.angular_velocity.z = x_hat_(BZ);
  bias.angular_velocity_covariance[0*3+0] = P_(BX,BX);
  bias.angular_velocity_covariance[1*3+1] = P_(BY,BY);
  bias.angular_velocity_covariance[2*3+2] = P_(BZ,BZ);

  bias_pub_.publish(bias);
  double ax(x_hat_(AX)), ay(x_hat_(AY)),az(x_hat_(AZ));
}


double mocapFilter::LPF(double yn, double un)
{
  return alpha_*yn+(1-alpha_)*un;
}

void mocapFilter::GPS_to_m(double* dlat, double* dlon, double* dx, double* dy)
{
  static const double R = 6371000.0; // radius of earth in meters
  double R_prime = cos(lat0_*M_PI/180.0)*R; // assumes you don't travel huge amounts

  // Converts latitude and longitude to meters
  // Be sure to de-reference pointers!
  (*dx) = sin((*dlat)*(M_PI/180.0))*R;
  (*dy) = sin((*dlon)*(M_PI/180.0))*R_prime;
}



} // namespace ekf


