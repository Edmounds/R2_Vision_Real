#ifndef COORDINATE_CALCULATOR_H
#define COORDINATE_CALCULATOR_H

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cmath>
#include <string>

/**
 * @brief 坐标计算器，负责计算篮框与相机间的距离和方位
 * 
 * 该类接收篮框位姿信息，计算相对距离、方向向量，
 * 并应用平滑滤波以提高稳定性。同时提供可视化功能。
 */
class CoordinateCalculator
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // 参数
    double hoop_rim_height_;        // 篮筐高度
    double camera_height_;          // 相机高度
    bool use_rim_height_correction_; // 是否使用篮筐高度校正
    double rim_z_offset_;           // 篮筐Z轴偏移
    
    // 篮筐在世界坐标系中的预期高度
    double expected_rim_world_z_;
    
    // ROS订阅者和发布者
    ros::Subscriber hoop_pose_sub_;  // 篮框位姿订阅者
    ros::Publisher distance_pub_;     // 距离发布者
    ros::Publisher distance_vector_pub_; // 距离向量发布者
    ros::Publisher adjusted_pose_pub_;   // 调整后位姿发布者
    ros::Publisher distance_marker_pub_; // 距离标记发布者
    
    // 上一次发布的距离，用于滤波
    double last_distance_x_;
    double last_distance_y_;
    double last_distance_z_;
    bool first_measurement_;
    double smoothing_factor_;

public:
    /**
     * @brief 构造函数，初始化计算器
     */
    CoordinateCalculator();
    
    /**
     * @brief 加载参数
     */
    void loadParameters();
    
    /**
     * @brief 篮框位姿回调函数
     * @param pose_msg 篮框位姿消息
     */
    void hoopPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose_msg);
    
    /**
     * @brief 可视化距离
     * @param stamp 时间戳
     * @param frame_id 坐标系ID
     * @param x X坐标
     * @param y Y坐标
     * @param z Z坐标
     */
    void visualizeDistance(const ros::Time& stamp, const std::string& frame_id,
                          double x, double y, double z);
};

#endif // COORDINATE_CALCULATOR_H