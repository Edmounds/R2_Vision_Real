#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <sensor_msgs/LaserScan.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cmath>

// 回调函数：处理 /tracked_pose 话题
void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg, ros::Publisher& target_pub, ros::NodeHandle& nh) {
  // 读取 point_of_interest 参数
  double point_x, point_y, point_z;
  nh.param("/point_of_interest/x", point_x, 0.0);
  nh.param("/point_of_interest/y", point_y, 0.0);
  nh.param("/point_of_interest/z", point_z, 0.0);

  // 提取机器人位姿（map 坐标系）
  double robot_x = msg->pose.position.x;
  double robot_y = msg->pose.position.y;
  double qx = msg->pose.orientation.x;
  double qy = msg->pose.orientation.y;
  double qz = msg->pose.orientation.z;
  double qw = msg->pose.orientation.w;

  // 计算 yaw 角（2D 场景）
  double yaw = std::atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz);

  // 计算感兴趣点相对于机器人的相对位置
  tf2::Transform map_to_robot;
  map_to_robot.setOrigin(tf2::Vector3(robot_x, robot_y, 0.0));
  map_to_robot.setRotation(tf2::Quaternion(qx, qy, qz, qw));
  tf2::Vector3 point_in_map(point_x, point_y, point_z);
  tf2::Transform robot_to_map = map_to_robot.inverse();
  tf2::Vector3 point_in_robot = robot_to_map * point_in_map;

  // 创建目标消息
  geometry_msgs::Pose2D target_msg;
  target_msg.x = point_in_robot.x();
  target_msg.y = point_in_robot.y();
  target_msg.theta = yaw;

  // 发布到 /target 话题
  target_pub.publish(target_msg);

  // 日志输出
  ROS_INFO_STREAM("Published /target: x=" << target_msg.x
                  << ", y=" << target_msg.y
                  << ", theta=" << target_msg.theta << " rad ("
                  << target_msg.theta * 180.0 / M_PI << " deg)");
}

void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg, ros::Publisher& target_pub, ros::NodeHandle& nh) {
  // 将获取到的激光点云转换为二值图像
  cv::Mat scan_image(1000, 1000, CV_8UC1, cv::Scalar(0));
  double angle_increment = msg->angle_increment;
  double angle_min = msg->angle_min;
  double range_min = msg->range_min;
  double range_max = msg->range_max;
  // ROS_INFO("Received LaserScan with %zu ranges, angle_min: %.2f, angle_max: %.2f, range_min: %.2f, range_max: %.2f",
  //          msg->ranges.size(), angle_min, msg->angle_max, range_min, range_max);

  for(size_t i = 0; i < msg->ranges.size(); ++i) {
    double range = msg->ranges[i];
    if (range < range_min || range > range_max) continue; // 忽略无效范围

    double angle = angle_min + i * angle_increment;
    int x = static_cast<int>(500 + range * 75 * std::cos(angle)); // 中心点为 (1000, 1000)
    int y = static_cast<int>(500 - range * 75 * std::sin(angle)); // Y轴向下

    if (x >= 0 && x < scan_image.cols && y >= 0 && y < scan_image.rows) {
      // scan_image.at<uchar>(y, x) = 255; // 设置像素为白色
      cv::circle(scan_image, cv::Point(x, y), 3, cv::Scalar(255), -1); // 绘制点
    }
  }

  // 进行霍夫变换检测直线
  std::vector<cv::Vec2f> lines;
  cv::HoughLines(scan_image, lines, 1, CV_PI / 180, 40);
  ROS_INFO("Detected %zu lines in the laser scan", lines.size());

  for (size_t j = 0; j < lines.size(); ++j) {
    float rho = lines[j][0];
    float theta = lines[j][1];

    bool erased = false;
    for (size_t k = 0; k < j; ++k) {
      float rho_prev = lines[k][0];
      float theta_prev = lines[k][1];

      if (abs(theta_prev - theta) < 0.5 and abs(rho_prev - rho) < 100) {
        lines.erase(lines.begin() + j);
        erased = true;
        break;
      }
    }
    if (erased) continue;

    double a = std::cos(theta);
    double b = std::sin(theta);
    double x0 = a * rho;
    double y0 = b * rho;
    cv::Point pt1(static_cast<int>(x0 + 1000 * (-b)), static_cast<int>(y0 + 1000 * a));
    cv::Point pt2(static_cast<int>(x0 - 1000 * (-b)), static_cast<int>(y0 - 1000 * a));
    cv::line(scan_image, pt1, pt2, cv::Scalar(255), 1, cv::LINE_AA);
  }

  cv::imshow("Laser Scan", scan_image);
  cv::waitKey(1);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "data_process_node");
  ros::NodeHandle nh;
  ros::Publisher target_pub = nh.advertise<geometry_msgs::Pose2D>("target", 10);
  ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(
      "tracked_pose", 10, boost::bind(poseCallback, _1, boost::ref(target_pub), boost::ref(nh)));
  ros::Subscriber scan_sub = nh.subscribe<sensor_msgs::LaserScan>(
      "scan", 10, boost::bind(scanCallback, _1, boost::ref(target_pub), boost::ref(nh)));
  ROS_INFO("Data process node started ===================================================================================================");
  ros::spin();
  return 0;
}