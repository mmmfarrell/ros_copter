#pragma once

#include <Eigen/Core>
#include <geometry/xform.h>

namespace roscopter
{

namespace ekf
{


class ErrorState
{
public:
    enum {
        DX = 0,
        DP = 0,
        DQ = 3,
        DV = 6,
        DBA = 9,
        DBG = 12,
        DBB = 15,
        DREF = 16,
        DGP = 17,
        DGV = 19,
        DGATT = 21,
        DGW = 22,
        DLMS = 23,
        MAX_LMS = 10,
        NDX = 23 + 3 * MAX_LMS,
        SIZE = 23 + 3 * MAX_LMS
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<double, SIZE, 1> arr;
    Eigen::Map<Vector6d> x;
    Eigen::Map<Eigen::Vector3d> p;
    Eigen::Map<Eigen::Vector3d> q;
    Eigen::Map<Eigen::Vector3d> v;
    Eigen::Map<Eigen::Vector3d> ba;
    Eigen::Map<Eigen::Vector3d> bg;
    double& bb; // bias for barometer measurements
    double& ref; // reference global altitude of NED frame
    Eigen::Map<Eigen::Vector2d> gp;
    Eigen::Map<Eigen::Vector2d> gv;
    double& gatt;
    double& gw;
    Eigen::Map<Eigen::Matrix<double, MAX_LMS, 3>> lms;

    ErrorState();
    ErrorState(const ErrorState& obj);
    ErrorState& operator=(const ErrorState& obj);
    ErrorState operator*(const double& s) const;
    ErrorState operator/(const double& s) const;
    ErrorState& operator*=(const double& s);
    ErrorState operator+(const ErrorState& obj) const;
    ErrorState operator-(const ErrorState& obj) const;
    ErrorState operator+(const Eigen::Matrix<double, SIZE, 1>& obj) const;
    ErrorState operator-(const Eigen::Matrix<double, SIZE, 1>& obj) const;
    ErrorState& operator+=(const Eigen::Matrix<double, SIZE, 1>& obj);
    ErrorState& operator-=(const Eigen::Matrix<double, SIZE, 1>& obj);
    ErrorState& operator+=(const ErrorState& obj);
    ErrorState& operator-=(const ErrorState& obj);

    static ErrorState Random()
    {
        ErrorState x;
        x.arr.setRandom();
        return x;
    }

    static ErrorState Zero()
    {
        ErrorState x;
        x.arr.setZero();
        return x;
    }
};

class State
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum {
      T = 0,
      X = 1, // for Xform access (p, q)
      P = 1,
      Q = 4,
      V = 8,
      BA = 11,
      BG = 14,
      BB = 17,
      REF = 18,
      GP = 19, // Goal position
      GV = 21, // Goal velocity
      GATT = 23, // Goal yaw attitude
      GW = 24, // Goal yaw angular velocity
      LMS = 25,
      MAX_LMS = 10,
      NX = 24 + 3 * MAX_LMS,
      A = 1 + NX,
      W = 1 + NX + 3,
      SIZE = 1 + NX + 6
  };
  Eigen::Matrix<double, SIZE, 1> arr;

  Eigen::Map<Vector6d> imu; // IMU measurement at current time
  Eigen::Map<Eigen::Vector3d> a;
  Eigen::Map<Eigen::Vector3d> w;

  double& t; // Time of current state
  xform::Xformd x;
  Eigen::Map<Eigen::Vector3d> p;
  quat::Quatd q;
  Eigen::Map<Eigen::Vector3d> v;
  Eigen::Map<Eigen::Vector3d> ba;
  Eigen::Map<Eigen::Vector3d> bg;
  double& bb; // barometer pressure bias
  double& ref; // reference global altitude of NED frame
  Eigen::Map<Eigen::Vector2d> gp;
  Eigen::Map<Eigen::Vector2d> gv;
  double& gatt;
  double& gw;
  Eigen::Map<Eigen::Matrix<double, MAX_LMS, 3>> lms;

  State();
  State(const State& other);
  State& operator=(const State& obj);

  static State Random()
  {
      State x;
      x.arr.setRandom();
      x.x = xform::Xformd::Random();
      return x;
  }

  static State Identity()
  {
      State out;
      out.x = xform::Xformd::Identity();
      out.v.setZero();
      out.ba.setZero();
      out.bg.setZero();
      out.bb = 0.;
      out.ref = 0.;
      out.gp.setZero();
      out.gv.setZero();
      out.gatt = 0.;
      out.gw = 0.;
      out.lms.setZero();

      return out;
  }

  State operator+(const ErrorState &delta) const;
  State operator+(const Eigen::Matrix<double, ErrorState::SIZE, 1> &delta) const;
  State& operator+=(const ErrorState &delta);
  State& operator+=(const Eigen::VectorXd& dx);
  ErrorState operator-(const State &x2) const;
};

typedef Eigen::Matrix<double, ErrorState::SIZE, ErrorState::SIZE> dxMat;
typedef Eigen::Matrix<double, ErrorState::SIZE, 1> dxVec;
typedef Eigen::Matrix<double, ErrorState::SIZE, 6> dxuMat;
typedef Eigen::Matrix<double, 6, 6> duMat;


class StateBuf
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Snapshot
    {
        State x;
        dxMat P;
    };

    std::vector<Snapshot, Eigen::aligned_allocator<Snapshot>> buf; // circular buffer
    int head;
    int tail;
    int size;

    StateBuf(int size);
    State &x();
    const State &x() const;
    dxMat &P();
    const dxMat &P() const;

    Snapshot& next();
    Snapshot& begin();

    void advance();
    bool rewind(double t);
};

}

}
