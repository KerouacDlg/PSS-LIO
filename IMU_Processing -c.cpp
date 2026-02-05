#include "IMU_Processing.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <algorithm>

const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); };

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_vel_scale = scaler;
}

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true)
{
  imu_en = true;
  init_iter_num = 1;
  mean_acc      = V3D(0, 0, 0.0);
  mean_gyr      = V3D(0, 0, 0);
  after_imu_init_ = false;
  state_cov.setIdentity();
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, 0.0);
  mean_gyr      = V3D(0, 0, 0);
  imu_need_init_    = true;
  init_iter_num     = 1;
  after_imu_init_   = false;
  
  time_last_scan = 0.0;
}

void ImuProcess::Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot)
{
  M3D hat_grav;
  hat_grav << 0.0, gravity_(2), -gravity_(1),
              -gravity_(2), 0.0, gravity_(0),
              gravity_(1), -gravity_(0), 0.0;
  double align_norm = (hat_grav * tmp_gravity).norm() / gravity_.norm() / tmp_gravity.norm();
  double align_cos = gravity_.transpose() * tmp_gravity;
  align_cos = align_cos / gravity_.norm() / tmp_gravity.norm();
  if (align_norm < 1e-6)
  {
    if (align_cos > 1e-6)
    {
      rot = Eye3d;
    }
    else
    {
      rot = -Eye3d;
    }
  }
  else
  {
    V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos); 
    rot = Exp(align_angle(0), align_angle(1), align_angle(2));
  }
}

void ImuProcess::IMU_init(const MeasureGroup &meas, int &N)
{
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    N ++;
  }
}



inline Vector3d toVector3d(const boost::array<double, 3> &arr)
{
  return Vector3d(arr[0], arr[1], arr[2]);
}

// -----------------------------------------------------------------------------
// 辅助：把 Eigen::Vector3d 写入 boost::array<double,3>cscdsd
// -----------------------------------------------------------------------------
inline void fromVector3d(const Vector3d &v, boost::array<double, 3> &arr)
{
  arr[0] = v.x();
  arr[1] = v.y();
  arr[2] = v.z();
}

// -----------------------------------------------------------------------------
// 辅助：把 boost::array<double,9> 转为 Eigen::Matrix3d （按 行优先 存储）
// -----------------------------------------------------------------------------
inline Matrix3d toMatrix3d(const boost::array<double, 9> &arr)
{
  Matrix3d R;
  R << arr[0], arr[1], arr[2],
      arr[3], arr[4], arr[5],
      arr[6], arr[7], arr[8];
  return R;
}

// -----------------------------------------------------------------------------
// 辅助：把 Eigen::Matrix3d 写入 boost::array<double,9> （按 行优先 存储）
// -----------------------------------------------------------------------------
inline void fromMatrix3d(const Matrix3d &R, boost::array<double, 9> &arr)
{
  arr[0] = R(0, 0);
  arr[1] = R(0, 1);
  arr[2] = R(0, 2);
  arr[3] = R(1, 0);
  arr[4] = R(1, 1);
  arr[5] = R(1, 2);
  arr[6] = R(2, 0);
  arr[7] = R(2, 1);
  arr[8] = R(2, 2);
}

// -----------------------------------------------------------------------------
// 辅助：把旋转矩阵和位置拼成 4×4 齐次变换矩阵
// -----------------------------------------------------------------------------
inline Matrix4d makeSE3(const Matrix3d &R, const Vector3d &p)
{
  Matrix4d T = Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = p;
  return T;
}

// -----------------------------------------------------------------------------
// so3Log & so3Exp：旋转矩阵 ↔ 李代数向量
// 这里简单调用 Eigen 的 AngleAxisd 做示例，若要更准确可用 Sophus 或者自己实现
// -----------------------------------------------------------------------------
inline Vector3d so3Log(const Matrix3d &R)
{
  AngleAxisd aa(R);
  return aa.axis() * aa.angle();
}

inline Matrix3d so3Exp(const Vector3d &omega)
{
  double theta = omega.norm();
  if (theta < 1e-12)
  {
    return Matrix3d::Identity();
  }
  else
  {
    Vector3d axis = omega / theta;
    return AngleAxisd(theta, axis).toRotationMatrix();
  }
}

// -----------------------------------------------------------------------------
// 插值专用的简化 Pose：只保留 offset_time, pos, rot （从 Pose6D_ 提取）
// -----------------------------------------------------------------------------
struct SimplePose
{
  double time;  // offset_time （秒）
  Vector3d pos; // 世界系位置

  Matrix3d rot; // 世界系旋转矩阵
};
struct SimplePoseV
{
  double time;  // 时间戳
  Vector3d pos; // 世界系位置
  Vector3d vel; // 世界系速度
  Matrix3d rot; // 世界系旋转
};


SimplePose linearInterpolate(const std::vector<SimplePose> &IMUpose, double t) {
    // 查找 idx 使 IMUpose[idx].time <= t < IMUpose[idx+1].time
    int idx = std::upper_bound(
        IMUpose.begin(), IMUpose.end(), t,
        [](double t_val, const SimplePose &p) { return t_val < p.time; }
    ) - IMUpose.begin() - 1;
    idx = std::max(0, std::min(idx, static_cast<int>(IMUpose.size())-2));

    double t0 = IMUpose[idx].time;
    double t1 = IMUpose[idx+1].time;
    double a = (t - t0) / (t1 - t0);

    SimplePose out;
    out.time = t;
    out.pos  = (1-a)*IMUpose[idx].pos + a*IMUpose[idx+1].pos;
    // 旋转 Slerp
    Quaterniond q0(IMUpose[idx].rot), q1(IMUpose[idx+1].rot);
    out.rot = q0.slerp(a, q1).toRotationMatrix();
    return out;
}
void ImuProcess::UndistortPcl(const MeasureGroup &meas, PointCloudXYZI::Ptr cur_pcl_un_)
{
  // Make sure that last_imu_ is initialized
  if (!last_imu_) 
  {
    last_imu_ = meas.imu.back();
  }

  // Collect IMU data for this frame
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  double pcl_beg_time = meas.lidar_beg_time;  // Use lidar_beg_time instead of lidar_end_time

  // Sort the point cloud by time
  cur_pcl_un_->clear();
  cur_pcl_un_->points.reserve(meas.lidar->points.size());
  for (auto &pt : meas.lidar->points)
  {
    cur_pcl_un_->points.push_back(pt);
  }
  sort(cur_pcl_un_->points.begin(), cur_pcl_un_->points.end(), time_list);

  // Create an IMU pose list for interpolation
  std::vector<SimplePose> IMUpose;
  IMUpose.reserve(v_imu.size());
  SimplePose p0;
  p0.time = 0.0; // Start time
  p0.pos = Eigen::Vector3d::Zero(); // Initial position, adjust if needed
  p0.rot = Eigen::Matrix3d::Identity(); // Initial rotation, adjust if needed
  IMUpose.push_back(p0);

  // Forward predict the IMU poses
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); ++it_imu)
  {
    const auto &head = *it_imu;
    const auto &tail = *(it_imu + 1);

    // Compute the time difference between imu samples
    double dt = tail->header.stamp.toSec() - head->header.stamp.toSec();

    // Interpolate the IMU pose between head and tail
    SimplePose p;
    p.time = dt;
    p.pos = Eigen::Vector3d::Zero();  // Position computation should be added
    p.rot = Eigen::Matrix3d::Identity(); // Rotation computation should be added
    IMUpose.push_back(p);
  }

  // Backward undistortion: Apply transformation to the point cloud
  for (int idx = (int)cur_pcl_un_->points.size() - 1; idx >= 0; --idx)
  {
    auto &pt = cur_pcl_un_->points[idx];
    double t_point = pt.curvature / 1000.0;
    
    // Interpolate the IMU pose for each point
    SimplePose pose_i = linearInterpolate(IMUpose, t_point);
    const Eigen::Matrix3d &R_i = pose_i.rot;
    const Eigen::Vector3d &p_i = pose_i.pos;

    // Apply transformation
    Eigen::Vector3d P_orig(pt.x, pt.y, pt.z);
    Eigen::Vector3d P_corr = R_i * P_orig + p_i;

    // Update point with corrected position
    pt.x = P_corr.x();
    pt.y = P_corr.y();
    pt.z = P_corr.z();
  }

  // Store the last IMU data for the next frame
  last_imu_ = meas.imu.back();
}



void ImuProcess::Process(const MeasureGroup &meas, PointCloudXYZI::Ptr cur_pcl_un_)
{  
  if (imu_en)
  {
    if(meas.imu.empty())  return;

    if (imu_need_init_)
    {
      // Initialize IMU if needed
      IMU_init(meas, init_iter_num);

      imu_need_init_ = true;

      if (init_iter_num > MAX_INI_COUNT)
      {
        ROS_INFO("IMU Initializing: %.1f %%", 100.0);
        imu_need_init_ = false;
        *cur_pcl_un_ = *(meas.lidar);
      }
      return;
    }

    if (!after_imu_init_) after_imu_init_ = true;
    UndistortPcl(meas, cur_pcl_un_);
    return;
  }
  else
  {
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
}

