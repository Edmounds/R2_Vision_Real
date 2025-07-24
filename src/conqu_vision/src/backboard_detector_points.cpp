#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/extract_indices.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>

// 全局变量
ros::Publisher marker_pub;
ros::Publisher filtered_cloud_pub;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterDepthDifference(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, 
    float min_edge_depth = 0.2,
    float max_surface_depth = 0.05,
    int num_neighbors = 5
)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // 检查点云是否为有序点云
    if (cloud->height == 1) {
        ROS_WARN("Point cloud is unorganized, cannot perform neighbor comparison");
        return filtered_cloud;
    }
    
    // 设置输出点云的基本信息
    filtered_cloud->header = cloud->header;
    filtered_cloud->is_dense = false;
    
    // ROS_INFO("Processing organized point cloud: %d x %d", cloud->width, cloud->height);
    
    // 遍历有序点云
    for (uint32_t v = num_neighbors; v < cloud->height - num_neighbors; ++v) {
        for (uint32_t u = num_neighbors; u < cloud->width - num_neighbors; ++u) {

            // 获取当前点
            pcl::PointXYZRGB current_point = cloud->at(u, v);
            
            // 检查当前点的有效性
            if (!std::isfinite(current_point.x) || !std::isfinite(current_point.y) || !std::isfinite(current_point.z)) {
                continue;
            }
            
            // 扫描相邻直线取最值
            float depth_diff_l = 0;
            float depth_diff_r = 255;
            float depth_diff_u = 255;
            float depth_diff_d = 0;
            
            // 扫描左侧直线 (u-1, v-5到v+5)
            for (int dv = -num_neighbors; dv <= num_neighbors; ++dv) {
                if (v + dv >= 0 && v + dv < cloud->height) {
                    pcl::PointXYZRGB left_point = cloud->at(u - 1, v + dv);
                    if (std::isfinite(left_point.x) && std::isfinite(left_point.y) && std::isfinite(left_point.z)) {
                        float diff = current_point.z - left_point.z;
                        depth_diff_l = std::max(depth_diff_l, diff);
                    }
                }
            }
            
            // 扫描右侧直线 (u+1, v-5到v+5)
            for (int dv = -num_neighbors; dv <= num_neighbors; ++dv) {
                if (v + dv >= 0 && v + dv < cloud->height) {
                    pcl::PointXYZRGB right_point = cloud->at(u + 1, v + dv);
                    if (std::isfinite(right_point.x) && std::isfinite(right_point.y) && std::isfinite(right_point.z)) {
                        float diff = current_point.z - right_point.z;
                        depth_diff_r = std::min(depth_diff_r, diff);
                    }
                }
            }
            
            // 扫描上侧直线 (u-5到u+5, v-1)
            for (int du = -num_neighbors; du <= num_neighbors; ++du) {
                if (u + du >= 0 && u + du < cloud->width) {
                    pcl::PointXYZRGB up_point = cloud->at(u + du, v - 1);
                    if (std::isfinite(up_point.x) && std::isfinite(up_point.y) && std::isfinite(up_point.z)) {
                        float diff = current_point.z - up_point.z;
                        depth_diff_u = std::min(depth_diff_u, diff);
                    }
                }
            }
            
            // 扫描下侧直线 (u-5到u+5, v+1)
            for (int du = -num_neighbors; du <= num_neighbors; ++du) {
                if (u + du >= 0 && u + du < cloud->width) {
                    pcl::PointXYZRGB down_point = cloud->at(u + du, v + 1);
                    if (std::isfinite(down_point.x) && std::isfinite(down_point.y) && std::isfinite(down_point.z)) {
                        float diff = current_point.z - down_point.z;
                        depth_diff_d = std::max(depth_diff_d, diff);
                    }
                }
            }

            // 输出调试信息
            // ROS_INFO("Point (%d, %d): Depth diff L: %.2f, R: %.2f, U: %.2f, D: %.2f",
            //           u, v, depth_diff_l, depth_diff_r, depth_diff_u, depth_diff_d);
            
            // 应用筛选条件 筛选出左下角的点
            if (depth_diff_l > min_edge_depth
            && depth_diff_u < max_surface_depth
            && depth_diff_r < max_surface_depth
            && depth_diff_d > min_edge_depth) {
                filtered_cloud->push_back(current_point);
            }
        }
    }
    
    // 更新点云尺寸信息
    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    
    return filtered_cloud;
}

float min_edge_depth, max_surface_depth;
int num_neighbors;

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
    // 检查原始ROS消息包含的字段
    // ROS_INFO("Point cloud fields:");
    // for (const auto& field : cloud_msg->fields) {
    //     ROS_INFO("  Field: %s, offset: %d, datatype: %d, count: %d", 
    //              field.name.c_str(), field.offset, field.datatype, field.count);
    // }

    // 转换ROS消息到PCL点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_msg, *input_cloud);
    
    if (input_cloud->empty()) {
        ROS_WARN("Received empty point cloud data");
        return;
    }
    
    // ROS_INFO("Got point cloud data(%zu)", input_cloud->size());
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud = 
        filterDepthDifference(input_cloud, min_edge_depth, max_surface_depth, num_neighbors);
    
    // ROS_INFO("Filtered point cloud size: %zu", filtered_cloud->size());
    
    // 发布筛选后的点云
    if (!filtered_cloud->empty()) {
        sensor_msgs::PointCloud2 output_cloud;
        pcl::toROSMsg(*filtered_cloud, output_cloud);
        output_cloud.header = cloud_msg->header;
        filtered_cloud_pub.publish(output_cloud);
    }
}

int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "backboard_detector_points");
    ros::NodeHandle nh("~");

    ROS_INFO("===== Point cloud detection start =====");

    nh.param("min_edge_depth", min_edge_depth, 0.2f);        // 默认值 0.2m
    nh.param("max_surface_depth", max_surface_depth, 0.05f);  // 默认值 0.05m
    nh.param("num_neighbors", num_neighbors, 5);

    ROS_INFO("Using parameters: min_edge_depth=%.2f, max_surface_depth=%.2f, num_neighbors=%d",
             min_edge_depth, max_surface_depth, num_neighbors);
    
    // 创建订阅者和发布者
    ros::Subscriber cloud_sub = nh.subscribe("/camera/depth_registered/points", 1, pointCloudCallback);
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("backboard_markers", 1);
    filtered_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_points", 1);
    
    // 进入循环
    ros::spin();
    
    return 0;
}