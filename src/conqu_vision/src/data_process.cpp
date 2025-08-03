#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
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

int main(int argc, char** argv) {
  ros::init(argc, argv, "data_process_node");
  ros::NodeHandle nh;
  ros::Publisher target_pub = nh.advertise<geometry_msgs::Pose2D>("target", 10);
  ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(
      "tracked_pose", 10, boost::bind(poseCallback, _1, boost::ref(target_pub), boost::ref(nh)));
  ros::spin();
  return 0;
}