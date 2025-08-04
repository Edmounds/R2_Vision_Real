#include <ros/ros.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/tf.h>
#include <cartographer_ros_msgs/FinishTrajectory.h>
#include <cartographer_ros_msgs/StartTrajectory.h>
#include <cstdio>

// 全局轨迹ID
int traj_id = 1;

// 初始位置回调函数
void init_pose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg)
{
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    double theta = tf::getYaw(msg->pose.pose.orientation);
    ros::NodeHandle nh;

    // 调用finish_trajectory服务结束当前轨迹
    ros::ServiceClient client_traj_finish = nh.serviceClient<cartographer_ros_msgs::FinishTrajectory>("finish_trajectory");
    cartographer_ros_msgs::FinishTrajectory srv_traj_finish;
    srv_traj_finish.request.trajectory_id = traj_id;
    if (client_traj_finish.call(srv_traj_finish))
    {
        ROS_INFO("调用finish_trajectory %d 成功！", traj_id);
    }
    else
    {
        ROS_ERROR("调用finish_trajectory服务失败！");
    }

    // 调用start_trajectory服务启动新轨迹
    ros::ServiceClient client_traj_start = nh.serviceClient<cartographer_ros_msgs::StartTrajectory>("start_trajectory");
    cartographer_ros_msgs::StartTrajectory srv_traj_start;
    srv_traj_start.request.configuration_directory = "/home/rc1/cartographer_ws/src/cartographer_ros/cartographer_ros/configuration_files";
    srv_traj_start.request.configuration_basename = "backpack_2d_localization.lua";
    srv_traj_start.request.use_initial_pose = 1;
    srv_traj_start.request.initial_pose = msg->pose.pose;
    srv_traj_start.request.relative_to_trajectory_id = 0;
    printf("&&&&&: %f__%f\n", srv_traj_start.request.initial_pose.position.x, srv_traj_start.request.initial_pose.position.y);
    if (client_traj_start.call(srv_traj_start))
    {
        ROS_INFO("调用start_trajectory %d 成功！", traj_id);
        traj_id++; // 轨迹ID自增
    }
    else
    {
        ROS_ERROR("调用start_trajectory服务失败！");
    }
}

int main(int argc, char **argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "initial_pose_node");
    ros::NodeHandle nh;

    // 订阅/initialpose话题
    ros::Subscriber pose_init_sub = nh.subscribe("/initialpose", 1, init_pose_callback);

    // 保持节点运行
    ros::spin();

    return 0;
}