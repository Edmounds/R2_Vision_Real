#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Point.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <cmath>

class BasketDetector
{
public:
  BasketDetector() : tf_buffer_(ros::Duration(10))
  {
  // 参数设置
  basket_height_ = 2.285;  // 篮板底部距地面高度(m)
  basket_length_ = 1.8;    // 篮板长度(m)
  basket_width_ = 0.05;    // 篮板底部边缘线宽度(m)，不是篮板高度
  height_tolerance_ = 0.05; // 减小高度容差，提高准确度
  size_tolerance_ = 0.2;   // 尺寸容差(m)
 
    // 设置TF监听器
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);

    // 订阅点云话题
    cloud_sub_ = nh_.subscribe("/r2/depth_camera/points", 1, &BasketDetector::cloudCallback, this);

    // 发布距离信息
    distance_pub_ = nh_.advertise<std_msgs::Float64>("/basket_distance", 10);
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
    // 1. 转换点云格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

     // 2. 转换坐标系到机器人基座坐标系 - 修正坐标系名称
    sensor_msgs::PointCloud2 transformed_cloud_msg;
    try {
      // 使用完整的坐标系名称
      tf_buffer_.transform(*cloud_msg, transformed_cloud_msg, "r2/base_link",
                          ros::Duration(0.5));
      pcl::fromROSMsg(transformed_cloud_msg, *cloud);
    } catch (tf2::TransformException& ex) {
      ROS_WARN("Coordinate transform failed: %s", ex.what());
      return;
    }

    // 3. 高度过滤 - 筛选出篮板底部高度附近的点
    pcl::PointCloud<pcl::PointXYZ>::Ptr height_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass_z;
    pass_z.setInputCloud(cloud);
    pass_z.setFilterFieldName("z");
    // 调整过滤范围，更集中在底部区域
    pass_z.setFilterLimits(basket_height_ - height_tolerance_, basket_height_ + 0.05);
    pass_z.filter(*height_filtered);

    if (height_filtered->empty()) {
      ROS_DEBUG("No points found near the bottom height of the backboard");
      return;
    }
    ROS_DEBUG("Number of points remaining after height filtering: %zu", height_filtered->points.size());

    // 4. 平面分割 - 找到篮板平面
    pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> plane_seg;
    plane_seg.setOptimizeCoefficients(true);
    plane_seg.setModelType(pcl::SACMODEL_PLANE);
    plane_seg.setMethodType(pcl::SAC_RANSAC);
    plane_seg.setDistanceThreshold(0.03);
    plane_seg.setInputCloud(height_filtered);
    plane_seg.segment(*plane_inliers, *plane_coefficients);

    if (plane_inliers->indices.size() == 0) {
      ROS_DEBUG("No plane found within the given height range");
      return;
    }

    // 5. 提取平面上的点
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract_plane;
    extract_plane.setInputCloud(height_filtered);
    extract_plane.setIndices(plane_inliers);
    extract_plane.setNegative(false);
    extract_plane.filter(*plane_cloud);

    // 6. 验证平面是否垂直（篮板应该是垂直的）
    float nx = plane_coefficients->values[0];
    float ny = plane_coefficients->values[1];
    float nz = plane_coefficients->values[2];
    float normal_magnitude = std::sqrt(nx*nx + ny*ny + nz*nz);
    
    // 垂直平面的法线应该几乎水平
    if (std::abs(nz / normal_magnitude) > 0.2) { // 允许20度偏差
      ROS_DEBUG("Detected plane is not vertical enough, unlikely to be the backboard");
      return;
    }

    // 7. 在平面中找出底部边缘
    // 找出平面点云中最低的点
    float min_height = std::numeric_limits<float>::max();
    for (const auto& point : plane_cloud->points) {
      if (point.z < min_height) {
        min_height = point.z;
      }
    }

    // 筛选出底部边缘的点
    pcl::PointCloud<pcl::PointXYZ>::Ptr bottom_edge_points(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : plane_cloud->points) {
      if (std::abs(point.z - min_height) < basket_width_) {
        bottom_edge_points->points.push_back(point);
      }
    }

    if (bottom_edge_points->empty()) {
      ROS_DEBUG("Failed to find bottom edge points on the backboard");
      return;
    }

    // 8. 在底部边缘点中进行线段拟合
    pcl::ModelCoefficients::Ptr line_coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr line_inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> line_seg;
    line_seg.setOptimizeCoefficients(true);
    line_seg.setModelType(pcl::SACMODEL_LINE);
    line_seg.setMethodType(pcl::SAC_RANSAC);
    line_seg.setDistanceThreshold(0.02);
    line_seg.setInputCloud(bottom_edge_points);
    line_seg.segment(*line_inliers, *line_coefficients);

    if (line_inliers->indices.size() == 0) {
      ROS_DEBUG("Failed to fit a line segment to the bottom edge points");
      return;
    }

    // 9. 提取线段上的点并计算长度
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract_line;
    extract_line.setInputCloud(bottom_edge_points);
    extract_line.setIndices(line_inliers);
    extract_line.setNegative(false);
    extract_line.filter(*line_cloud);

    // 找出线段的端点
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*line_cloud, min_pt, max_pt);

    // 计算线段长度
    float line_length = std::sqrt(std::pow(max_pt.x - min_pt.x, 2) + 
                                 std::pow(max_pt.y - min_pt.y, 2));

    if (std::abs(line_length - basket_length_) > size_tolerance_) {
      ROS_DEBUG("Bottom edge line length (%.2f) does not match expected backboard length (%.2f±%.2f)",
               line_length, basket_length_, size_tolerance_);
      return;
    }

    // 10. 计算机器人到底部边缘线的距离
    // 计算边缘线的中点
    pcl::PointXYZ center_point;
    center_point.x = (min_pt.x + max_pt.x) / 2;
    center_point.y = (min_pt.y + max_pt.y) / 2;
    center_point.z = (min_pt.z + max_pt.z) / 2;

    // 计算水平距离 (XY平面)
    float distance = std::sqrt(std::pow(center_point.x, 2) + 
                          std::pow(center_point.y, 2));
                          
    // 打印更详细的调试信息
    ROS_INFO("Bottom edge center point coordinates (r2/base_link frame): x=%.2f, y=%.2f, z=%.2f", 
        center_point.x, center_point.y, center_point.z);

    // 11. 发布距离信息
    std_msgs::Float64 distance_msg;
    distance_msg.data = distance;
    distance_pub_.publish(distance_msg);

    ROS_INFO("Horizontal distance from robot to bottom edge of backboard: %.2f m", distance);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher distance_pub_;

  tf2_ros::Buffer tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  float basket_height_;  // 篮框底部距地面高度
  float basket_length_;  // 篮框长度
  float basket_width_;   // 篮框宽度
  float height_tolerance_; // 高度容差
  float size_tolerance_;  // 尺寸容差
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "hoop_detector");
  BasketDetector detector;
  ros::spin();
  return 0;
}
