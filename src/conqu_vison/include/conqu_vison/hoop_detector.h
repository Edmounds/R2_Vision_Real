#ifndef HOOP_DETECTOR_H
#define HOOP_DETECTOR_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/common.h>

/**
 * @brief 篮框检测器，负责从点云中识别篮框
 * 
 * 该类处理来自深度相机的点云数据，通过一系列处理步骤
 * 检测篮框并发布其位置信息。
 */
class HoopDetector
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // 参数
    double hoop_length_;
    double hoop_width_;
    double hoop_bottom_height_;
    double hoop_top_height_;
    double hoop_rim_height_;
    double camera_height_;
    
    double height_filter_min_;
    double height_filter_max_;
    double ransac_distance_threshold_;
    double length_tolerance_;
    double angle_tolerance_;
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    
    // ROS订阅者和发布者
    ros::Subscriber point_cloud_sub_;
    ros::Publisher filtered_cloud_pub_;
    ros::Publisher hoop_pose_pub_;
    ros::Publisher visualization_pub_;

public:
    /**
     * @brief 构造函数，初始化检测器
     */
    HoopDetector();
    
    /**
     * @brief 加载参数
     */
    void loadParameters();
    
    /**
     * @brief 点云回调函数
     * @param cloud_msg 点云消息
     */
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    
    /**
     * @brief 根据高度过滤点云
     * @param cloud 输入点云
     * @return 过滤后的点云
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr filterByHeight(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    /**
     * @brief 去除点云噪声
     * @param cloud 输入点云
     * @return 去噪后的点云
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr removeNoise(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    /**
     * @brief 检测边缘
     * @param cloud 输入点云
     * @return 检测到的边缘系数
     */
    std::vector<pcl::ModelCoefficients> detectEdges(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    /**
     * @brief 验证几何形状
     * @param edges 检测到的边缘
     * @param hoop_pose 计算出的篮框位姿
     * @return 是否成功检测到篮框
     */
    bool validateGeometry(const std::vector<pcl::ModelCoefficients>& edges, 
                         geometry_msgs::PoseStamped& hoop_pose);
    
    /**
     * @brief 可视化篮框
     * @param pose 篮框位姿
     */
    void visualizeHoop(const geometry_msgs::PoseStamped& pose);
};

#endif